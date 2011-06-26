#include <QFileDialog>
#include <QSound>
#include <QMessageBox>

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "processingthread.h"
#include "irisvideocapture.h"
#include "tools.h"

extern ProcessingThread PROCESSING_THREAD;
extern IrisVideoCapture IRIS_VIDEO_CAPTURE;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
	ui(new Ui::MainWindow),
	lastFocusScores(1000, 0),
	debugDialog(this)
{
	this->ui->setupUi(this);

	this->ui->chkGuardarVideo->setChecked(!::IRIS_VIDEO_CAPTURE.isPaused());
	QObject::connect(this->ui->chkGuardarVideo, SIGNAL(stateChanged(int)), &::IRIS_VIDEO_CAPTURE, SLOT(setPause(int)));

	qRegisterMetaType<VideoProcessor>("VideoProcessor");
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::slotFrameAvailable(const ColorImage& /*frame*/)
{
}

void MainWindow::slotFrameProcessed(const VideoProcessor& videoProcessor)
{
	VideoProcessor::VideoStatus status = videoProcessor.lastStatus;

	this->lastProcessedFrame = videoProcessor.lastFrame;		// Note: this does not make a copy (faster)
	this->ui->video->slotFrameProcessed(videoProcessor);

	mostrarEnfoque(videoProcessor.lastFocusScore, videoProcessor.parameters.focusThreshold, videoProcessor.lastFrame.cols);

	int irisScore = ((status >= VideoProcessor::FOCUSED_NO_IRIS) ? videoProcessor.lastIrisQuality : 0);
	this->ui->irisScore->setValue(irisScore);
	this->ui->statusBar->showMessage(MainWindow::statusToString(status));
}

void MainWindow::slotGotTemplate(const VideoProcessor& videoProcessor)
{
	//this->lastTemplate = videoProcessor.getTemplate();
	this->lastTemplate = videoProcessor.getCapturedTemplate();
	this->lastIrisFrameSegmentation = videoProcessor.getBestTemplateSegmentation();
	this->lastIrisFrame = videoProcessor.getBestTemplateFrame().clone();
	this->lastCaptureBurst = videoProcessor.captureBurst;

	this->showTemplateImage();

	if (this->ui->autoIdentify->isChecked()) {
		this->identifyTemplate(this->lastTemplate, this->lastIrisFrame, this->lastIrisFrameSegmentation);
	}
}

void MainWindow::showTemplateImage()
{
	cvtColor(this->lastIrisFrame, this->decoratedFrame, CV_GRAY2RGB);

	this->decorator.setDrawingColors();
	this->decorator.lineWidth = 1;
	this->decorator.drawSegmentationResult(this->decoratedFrame, this->lastIrisFrameSegmentation);
	this->decorator.drawEncodingZone(this->decoratedFrame, this->lastIrisFrameSegmentation);

	// Muestra el frame decorado en la ventana de la imagen capturada. Para esto, hay que resizearlo
	cv::resize(this->decoratedFrame, this->resizedFrame, Size(320*1.2, 240*1.2));
	this->decorator.drawTemplate(this->resizedFrame, this->lastTemplate);

	Mat region = this->resizedFrame(Range(this->resizedFrame.rows-30, this->resizedFrame.rows), Range::all());
	//decorator.superimposeImage(this->lastIrisFrame, region, this->lastIrisFrameSegmentation);

	this->ui->capturedImage->showImage(this->resizedFrame);
}

void MainWindow::on_btnIdentificar_clicked()
{
	if (this->lastIrisFrame.empty()) return;				// Sólo hacer identificación si hay un iris

	this->identifyTemplate(this->lastTemplate, this->lastIrisFrame, this->lastIrisFrameSegmentation, this->lastCaptureBurst);
}

void MainWindow::on_btnRegistrar_clicked()
{
	if (this->lastIrisFrame.empty()) return;				// Sólo registrar si hay un iris

	this->registerTemplate(this->lastTemplate, this->lastIrisFrame, this->lastIrisFrameSegmentation, this->lastCaptureBurst);
}

void MainWindow::on_btnGuardarImagen_clicked()
{
	int ans = QMessageBox::question(this, "Guardar imagen", "Decorar la imagen a guardar?", QMessageBox::Yes, QMessageBox::No);
	if (ans == QMessageBox::Yes) {
		ColorImage decorated;
		cvtColor(this->lastIrisFrame, decorated, CV_GRAY2BGR);
		this->decorator.drawSegmentationResult(decorated, this->lastIrisFrameSegmentation);
		this->decorator.drawTemplate(decorated, this->lastTemplate);
		ofrecerGuardarImagen(decorated);
	} else {
		ofrecerGuardarImagen(this->lastIrisFrame);
	}
}

void MainWindow::on_btnCapturar_clicked()
{
	ofrecerGuardarImagen(this->lastProcessedFrame);
}

void MainWindow::ofrecerGuardarImagen(const Image& imagen)
{
	if (imagen.empty()) return;

	Mat copiaImagen = imagen.clone();			// Hago una copia porque la imagen se puede actualizar desde otro thread
	QString filename = QFileDialog::getSaveFileName(this, "Guardar imagen...");
	if (!filename.isEmpty()) {
		imwrite(filename.toStdString(), copiaImagen);
	}
}

void MainWindow::on_btnForzarRegistracion_clicked()
{
}

void MainWindow::identifyTemplate(const IrisTemplate& irisTemplate, const GrayscaleImage& image, const SegmentationResult& segmentationResult, horus::VideoProcessor::CaptureBurst captureBurst)
{
	if (this->matchingDialog.isVisible()) return;

	this->matchingDialog.doMatch(irisTemplate, image, segmentationResult, captureBurst);
}

void MainWindow::registerTemplate(const IrisTemplate& irisTemplate, const GrayscaleImage& image, const SegmentationResult& segmentationResult, horus::VideoProcessor::CaptureBurst captureBurst)
{
	if (this->registerDialog.isVisible()) return;

	this->registerDialog.doRegister(irisTemplate, image, segmentationResult, captureBurst);
}


void MainWindow::mostrarEnfoque(double enfoque, double threshold, int width)
{
	this->lastFocusScores.pop_front();
	this->lastFocusScores.push_back(enfoque);

	if (this->imagenEnfoque.empty()) {
		this->imagenEnfoque.create(Size(width, 20));
	}

	decorator.drawFocusScores(this->imagenEnfoque, this->lastFocusScores, Rect(0, 0, this->imagenEnfoque.cols, this->imagenEnfoque.rows), threshold);
	this->ui->animacionEnfoque->showImage(this->imagenEnfoque);
	this->ui->focusScore->setValue(enfoque);
}

void MainWindow::on_btnForzarIdentificacion_clicked()
{
	if (this->lastProcessedFrame.empty()) return;

	cvtColor(this->lastProcessedFrame, this->lastIrisFrame, CV_BGR2GRAY);

	this->lastIrisFrameSegmentation = ::PROCESSING_THREAD.videoProcessor.segmentator.segmentImage(this->lastIrisFrame);
	this->lastTemplate = ::PROCESSING_THREAD.videoProcessor.irisEncoder.generateTemplate(this->lastIrisFrame, this->lastIrisFrameSegmentation);

	this->showTemplateImage();

	this->identifyTemplate(this->lastTemplate, this->lastIrisFrame, this->lastIrisFrameSegmentation);
}

QString MainWindow::statusToString(VideoProcessor::VideoStatus status)
{
	switch (status) {
	case VideoProcessor::UNKNOWN_ERROR:
		return QString("Error en tiempo de ejecución");
	case VideoProcessor::UNPROCESSED:
		return QString("Esperando...");
	case VideoProcessor::DEFOCUSED:
		return QString("Video desenfocado");
	case VideoProcessor::INTERLACED:
		return QString("Frame entrelazado");
	case VideoProcessor::NO_EYE:
		return QString("Ojo no detectado");
	case VideoProcessor::FOCUSED_NO_IRIS:
		return QString("Iris no detectado");
	case VideoProcessor::IRIS_LOW_QUALITY:
		return QString("Baja calidad de iris");
	case VideoProcessor::IRIS_TOO_CLOSE:
		return QString("Iris demasiado cerca");
	case VideoProcessor::IRIS_TOO_FAR:
		return QString("Iris demasiado lejos");
	case VideoProcessor::FOCUSED_IRIS:
		return QString("Iris obtenido");
	case VideoProcessor::BAD_TEMPLATE:
		return QString("Baja calidad de template");
	case VideoProcessor::FINISHED_CAPTURE:
		return QString("Captura finalizada");
	case VideoProcessor::GOT_TEMPLATE:
		return QString("Template obtenido");
	}

	return QString("");
}

void MainWindow::on_debugWindow_clicked()
{
	this->debugDialog.open();
}
