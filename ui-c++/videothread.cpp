#include "videothread.h"

VideoThread::VideoThread(int cam) :
	_cam(cam)
{
}

void VideoThread::run()
{
	//_cap.open(_cam);
	//_cap.open("/home/marcelo/iris/BBDD/Videos/norberto1/20080501-230608.mpg");
	//_cap.open("/home/marcelo/iris/BBDD/Videos/marta1/20080702-232946.mpg");
	_cap.open("/home/marcelo/iris/BBDD/Videos/bursztyn1/20080501-230748.mpg");
	//_cap.open("/home/marcelo/iris/BBDD/Videos/marcelo1/marcelo1.mpg");

	if (!_cap.isOpened()) {
		qDebug() << "No se pudo inicializar video";
		return;
	}

	qDebug() << "Thread de video inicializado";
	_stop = false;

	while (!_stop) {
		_cap >> _frame;

		if (_frame.empty()) break;		// Fin del video (por algún motivo)

		flip(_frame, _frame, 1);		// El flip es para que el video no salga al revés

		emit(signalFrameAvailable(_frame));
		msleep(30);
	}

	_cap.release();
}
