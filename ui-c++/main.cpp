#include <QtGui/QApplication>
#include <QMetaType>

#include "mainwindow.h"
#include "videothread.h"
#include "processingthread.h"


int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	MainWindow w;
	w.show();

	// Inicializacion
	qRegisterMetaType<Mat>("Mat");
	qRegisterMetaType<VideoProcessor>("VideoProcessor");

	VideoThread irisVideoThread(0);
	ProcessingThread processingThread;
	QObject::connect(&irisVideoThread, SIGNAL(signalFrameAvailable(Mat)), &w, SLOT(slotFrameAvailable(Mat)));
	QObject::connect(&irisVideoThread, SIGNAL(signalFrameAvailable(Mat)), &processingThread, SLOT(slotProcessFrame(Mat)));
	QObject::connect(&processingThread, SIGNAL(signalFrameProcessed(VideoProcessor)), &w, SLOT(slotFrameProcessed(VideoProcessor)));
	irisVideoThread.start();

	// Ejecución
	int res = a.exec();

	// Terminación
	irisVideoThread.stop();
	irisVideoThread.wait();

	return res;
}
