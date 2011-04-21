#include "videothread.h"
#include <QDebug>

VideoThread::VideoThread(int cam) :
	_cap(cam)
{
}

void VideoThread::run()
{
	if (!_cap.isOpened()) {
		qDebug() << "No se pudo inicializar video";
		return;
	}

	qDebug() << "Thread de video inicializado";
	_stop = false;

	while (!_stop) {
		_cap >> _frame;

		emit(signalFrameAvailable(_frame));
	}

	_cap.release();
}
