#include "debugdialog.h"
#include "ui_debugdialog.h"
#include "processingthread.h"


extern ProcessingThread PROCESSING_THREAD;

DebugDialog::DebugDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DebugDialog)
{
	qRegisterMetaType<VideoProcessor>("VideoProcessor");
    ui->setupUi(this);
}

DebugDialog::~DebugDialog()
{
    delete ui;
}

void DebugDialog::open()
{
	QObject::connect(&PROCESSING_THREAD, SIGNAL(signalFrameProcessed(VideoProcessor)), this, SLOT(slotFrameProcessed(VideoProcessor)), Qt::BlockingQueuedConnection);

	PROCESSING_THREAD.videoProcessor.parameters.pauseAfterCapture = false;

	QDialog::open();
}


void DebugDialog::done(int r)
{
	QDialog::done(r);

	PROCESSING_THREAD.videoProcessor.parameters.pauseAfterCapture = true;

	QObject::disconnect(&PROCESSING_THREAD, SIGNAL(signalFrameProcessed(VideoProcessor)), this, SLOT(slotFrameProcessed(VideoProcessor)));
}

void DebugDialog::slotFrameProcessed(const VideoProcessor& videoProcessor)
{
	VideoProcessor::VideoStatus status = videoProcessor.lastStatus;

	if (status >= VideoProcessor::FOCUSED_NO_IRIS) {
		Image similarityImage;
		cvtColor(videoProcessor.segmentator.pupilSegmentator.similarityImage, similarityImage, CV_GRAY2BGR);
		this->ui->similarityImage->showImage(similarityImage);

		GrayscaleImage adjustmentRing = videoProcessor.segmentator.pupilSegmentator.adjustmentRing.clone();
		GrayscaleImage adjustmentRingGradient = horus::tools::normalizeImage(videoProcessor.segmentator.pupilSegmentator.adjustmentRingGradient);
		const Mat1f& adjustmentSnake = videoProcessor.segmentator.pupilSegmentator.adjustmentSnake;

		int delta = adjustmentRingGradient.rows * 0.1;

		for (int x = 0; x < adjustmentSnake.cols; x++) {
			float y = adjustmentSnake(0, x);
			adjustmentRing(y, x) = 255;

			adjustmentRingGradient(y, x) = 255;
			if (y-delta >= 0) adjustmentRingGradient(y-delta, x) = 255;
			if (y+delta < adjustmentRingGradient.rows) adjustmentRingGradient(y+delta, x) = 255;
		}
		this->ui->debug1->showImage(adjustmentRing);
		assert(adjustmentRingGradient.isContinuous());
		this->ui->debug2->showImage(adjustmentRingGradient);
	}
}
