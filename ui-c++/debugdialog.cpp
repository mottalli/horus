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
	}
}
