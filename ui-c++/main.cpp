#include <QtGui/QApplication>
#include <QMetaType>
#include <QDebug>

#include "mainwindow.h"
#include "videothread.h"
#include "processingthread.h"
#include "sqlite3irisdatabase.h"
#include "irisvideocapture.h"

/******* Globales *******/

const string pathBase = "/home/marcelo/iris/horus/base-iris";

SQLite3IrisDatabase DB(pathBase);
VideoThread IRIS_VIDEO_THREAD(1);
ProcessingThread PROCESSING_THREAD;
IrisVideoCapture IRIS_VIDEO_CAPTURE(pathBase);

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	MainWindow w;
	w.show();

	/******* Seteo parámetros *****/
	PROCESSING_THREAD.videoProcessor.parameters.doEyeDetect = false;
	PROCESSING_THREAD.videoProcessor.segmentator.pupilSegmentator.parameters.avoidPupilReflection = false;

	/******* Inicializacion *******/
	qRegisterMetaType<Mat>("Mat");
	qRegisterMetaType<ColorImage>("ColorImage");
	qRegisterMetaType<GrayscaleImage>("GrayscaleImage");
	qRegisterMetaType<VideoProcessor>("VideoProcessor");
	qRegisterMetaType<IrisTemplate>("IrisTemplate");

	QObject::connect(&IRIS_VIDEO_THREAD, SIGNAL(signalFrameAvailable(ColorImage)), &PROCESSING_THREAD, SLOT(slotProcessFrame(ColorImage)), Qt::BlockingQueuedConnection);
	QObject::connect(&PROCESSING_THREAD, SIGNAL(signalFrameProcessed(VideoProcessor)), &w, SLOT(slotFrameProcessed(VideoProcessor)), Qt::BlockingQueuedConnection);
	QObject::connect(&PROCESSING_THREAD, SIGNAL(signalGotTemplate(VideoProcessor)), &w, SLOT(slotGotTemplate(VideoProcessor)), Qt::BlockingQueuedConnection);
	QObject::connect(&PROCESSING_THREAD, SIGNAL(signalFrameProcessed(VideoProcessor)), &IRIS_VIDEO_CAPTURE, SLOT(slotFrameProcessed(VideoProcessor)));

	IRIS_VIDEO_THREAD.start();

	/******* Ejecución *******/
	int res = a.exec();

	/******* Terminación *******/

	// Si no se hace esto, hay un deadlock al salir
	QObject::disconnect(&IRIS_VIDEO_THREAD, SIGNAL(signalFrameAvailable(ColorImage)), &PROCESSING_THREAD, SLOT(slotProcessFrame(ColorImage)));
	QObject::disconnect(&PROCESSING_THREAD, SIGNAL(signalFrameProcessed(VideoProcessor)), &w, SLOT(slotFrameProcessed(VideoProcessor)));
	QObject::disconnect(&PROCESSING_THREAD, SIGNAL(signalGotTemplate(VideoProcessor)), &w, SLOT(slotGotTemplate(VideoProcessor)));
	QObject::disconnect(&PROCESSING_THREAD, SIGNAL(signalFrameProcessed(VideoProcessor)), &IRIS_VIDEO_CAPTURE, SLOT(slotFrameProcessed(VideoProcessor)));

	IRIS_VIDEO_THREAD.stop();
	qDebug() << "Esperando fin de video...";
	IRIS_VIDEO_THREAD.wait();
	qDebug() << "Fin de video.";

	return res;
}
