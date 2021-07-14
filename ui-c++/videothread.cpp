#include "videothread.h"
#include "drivers/ueyedriver.hpp"

VideoThread::VideoThread(int cam) :
    _cam(cam)
{
    _driver = new horus::UEyeVideoDriver();
    _driver->initializeCamera(cam);
}

VideoThread::~VideoThread()
{
    delete _driver;
}

void VideoThread::run()
{
#if 0
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
#endif

    auto processFrame = [&](ColorImage& frame) {
        flip(frame, frame, 1);
        emit(signalFrameAvailable(frame));
        msleep(10);

        if (_stop)
            _driver->stopCaptureThread();
    };

    std::thread captureThread = _driver->startCaptureThread(processFrame);
    captureThread.join();
}
