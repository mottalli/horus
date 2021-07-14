#include <QtGui/QApplication>
#include <QMetaType>
#include <QDebug>
#include <boost/program_options.hpp>
#include <fstream>

#include "mainwindow.h"
#include "videothread.h"
#include "processingthread.h"
#include "sqlite3irisdatabase.h"
#include "irisvideocapture.h"

/******* Globales *******/

const string pathBase = "/home/marcelo/iris/horus/base-iris";
//const string pathBase = "c:/Users/Marcelo/Desktop/horus/base-iris";

SQLite3IrisDatabase DB(pathBase);
VideoThread IRIS_VIDEO_THREAD;
ProcessingThread PROCESSING_THREAD;
IrisVideoCapture IRIS_VIDEO_CAPTURE(pathBase);

namespace options = boost::program_options;

void parseOptions(int argc, char** argv);

int main(int argc, char *argv[])
{
	/******* Inicializo parámetros *****/
	parseOptions(argc, argv);


	/******* Inicializo ventana principal *****/
	QApplication a(argc, argv);
	MainWindow w;
	w.show();

	/******* Seteo parámetros *****/
	PROCESSING_THREAD.videoProcessor.parameters.doEyeDetect = false;
	//PROCESSING_THREAD.videoProcessor.segmentator.pupilSegmentator.parameters.avoidPupilReflection = false;

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

/******* Parseo parámetros *****/

void parseOptions(int argc, char** argv)
{
	options::options_description desc("Opciones");
	desc.add_options()
			("video,v", options::value<int>()->default_value(0), "Numero de dispositivo de video")
			("help", "Ayuda")
	;

	try {
		options::variables_map vm;
		options::store(options::parse_command_line(argc, argv, desc), vm);

		// Read from "settings.ini" (if exists)
		string configFilename = "settings.ini";
		try {
			if (filesystem::is_regular_file(configFilename)) {
				ifstream input(configFilename.c_str());
				qDebug() << "Leyendo configuración de" << configFilename.c_str();
				options::store(options::parse_config_file(input, desc), vm);
			}
		} catch (...) {
			qDebug() << "Error leyendo datos de" << configFilename.c_str();
		}

		options::notify(vm);

		if (vm.count("help")) {
			cout << desc;
			exit(0);
		}

		IRIS_VIDEO_THREAD.setCapture(vm["video"].as<int>());
	} catch (options::error ex) {
		cout << ex.what() << endl;
		cout << desc;
		exit(1);
	}
}
