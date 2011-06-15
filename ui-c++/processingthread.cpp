#include "processingthread.h"

ProcessingThread::ProcessingThread(QObject *parent) :
    QThread(parent)
{
}

void ProcessingThread::run()
{
	Timer t;
	VideoProcessor::VideoStatus status = this->videoProcessor.processFrame(_frame);
	signalFrameProcessed(this->videoProcessor);

	//qDebug() << "FP: " << t.elapsed();

	if (status == VideoProcessor::GOT_TEMPLATE) {
		signalGotTemplate(this->videoProcessor);
	}
}

void ProcessingThread::slotProcessFrame(const ColorImage& frame)
{
	_frame = frame.clone();
	start();
}

