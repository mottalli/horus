#include <QtGui/QApplication>
#include <QMetaType>
#include <QDebug>

#include "mainwindow.h"
#include "videothread.h"
#include "processingthread.h"
#include "sqlite3irisdatabase.h"

SQLite3IrisDatabase DB("/home/marcelo/Documents/Programacion/horus/ui-python/_base");


VideoThread irisVideoThread(0);
ProcessingThread processingThread;

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	MainWindow w;
	w.show();

	/******* Inicializacion *******/
	qRegisterMetaType<Mat>("Mat");
	qRegisterMetaType<VideoProcessor>("VideoProcessor");
	qRegisterMetaType<IrisTemplate>("IrisTemplate");

	//QObject::connect(&irisVideoThread, SIGNAL(signalFrameAvailable(Mat)), &w, SLOT(slotFrameAvailable(Mat)));
	QObject::connect(&irisVideoThread, SIGNAL(signalFrameAvailable(Mat)), &processingThread, SLOT(slotProcessFrame(Mat)), Qt::BlockingQueuedConnection);
	QObject::connect(&processingThread, SIGNAL(signalFrameProcessed(VideoProcessor)), &w, SLOT(slotFrameProcessed(VideoProcessor)), Qt::BlockingQueuedConnection);
	QObject::connect(&processingThread, SIGNAL(signalGotTemplate(VideoProcessor)), &w, SLOT(slotGotTemplate(VideoProcessor)), Qt::BlockingQueuedConnection);
	irisVideoThread.start();

	/******* Ejecución *******/
	int res = a.exec();

	/******* Terminación *******/

	// Si no se hace esto, hay un deadlock al salir
	QObject::disconnect(&irisVideoThread, SIGNAL(signalFrameAvailable(Mat)), &processingThread, SLOT(slotProcessFrame(Mat)));
	QObject::disconnect(&processingThread, SIGNAL(signalFrameProcessed(VideoProcessor)), &w, SLOT(slotFrameProcessed(VideoProcessor)));
	QObject::disconnect(&processingThread, SIGNAL(signalGotTemplate(VideoProcessor)), &w, SLOT(slotGotTemplate(VideoProcessor)));

	irisVideoThread.stop();
	irisVideoThread.wait();

	return res;
}
