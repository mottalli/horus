#include "videothread.h"

VideoThread::VideoThread(int cam) :
	_cam(cam)
{
}

void VideoThread::run()
{
	bool use_avi = false;
	if (use_avi) {
		_cap.open("/home/marcelo/iris/BBDD/Videos/norberto1/20080501-230608.mpg");
		//_cap.open("/home/marcelo/iris/BBDD/Videos/marta1/20080702-232946.mpg");
		//_cap.open("/home/marcelo/iris/BBDD/Videos/bursztyn1/20080501-230748.mpg");
		//_cap.open("/home/marcelo/iris/BBDD/Videos/marcelo1/marcelo1.mpg");
	} else {
		qDebug() << "Abriendo dispositivo de video" << _cam;
		_cap.open(_cam);
		qDebug() << _cap.isOpened();
		_cap.set(CV_CAP_PROP_FRAME_WIDTH, 720);
		_cap.set(CV_CAP_PROP_FRAME_HEIGHT, 576);
	}

	if (!_cap.isOpened()) {
		qDebug() << "No se pudo inicializar video";
		return;
	}

	qDebug() << "Thread de video inicializado";
	_stop = false;

	while (!_stop) {
		_cap >> _frame;

		if (_frame.empty()) break;		// Fin del video (por algún motivo)

		flip(_frame, _frame, 1);		// El flip es para que el video no salga al revés (es anti-intuitivo para los usuarios)

		// Extraigo una sub-ventana porque los bordes suelen venir negros
		Mat subwindow = _frame(Range(30, _frame.rows-20), Range(18, _frame.cols-70));

		emit(signalFrameAvailable(subwindow));
		//emit(signalFrameAvailable(_frame));
		msleep(10);
	}

	_cap.release();
}
