#include "processingthread.h"

ProcessingThread::ProcessingThread(QObject *parent) :
    QThread(parent)
{
}

void ProcessingThread::run()
{
	_videoProcessor.processFrame(_frame);
	signalFrameProcessed(_videoProcessor);
}

void ProcessingThread::slotProcessFrame(const Mat& frame)
{
	_frame = frame;
	start();
}
