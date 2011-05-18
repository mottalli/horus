#include "processingthread.h"

ProcessingThread::ProcessingThread(QObject *parent) :
    QThread(parent)
{
}

void ProcessingThread::run()
{
	this->videoProcessor.parameters.doEyeDetect = false;
	VideoProcessor::VideoStatus status = this->videoProcessor.processFrame(_frame);
	signalFrameProcessed(this->videoProcessor);

	if (status == VideoProcessor::GOT_TEMPLATE) {
		signalGotTemplate(this->videoProcessor);
	}
}

void ProcessingThread::slotProcessFrame(const ColorImage& frame)
{
	_frame = frame.clone();
	start();
}

