#include "processingthread.h"

ProcessingThread::ProcessingThread(QObject *parent) :
    QThread(parent)
{
}

void ProcessingThread::run()
{
	VideoProcessor::VideoStatus status = _videoProcessor.processFrame(_frame);
	signalFrameProcessed(_videoProcessor);

	if (status == VideoProcessor::GOT_TEMPLATE) {
		signalGotTemplate(_videoProcessor);
	}
}

void ProcessingThread::slotProcessFrame(const Mat& frame)
{
	_frame = frame.clone();
	start();
}

