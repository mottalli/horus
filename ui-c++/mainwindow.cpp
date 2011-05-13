#include <QDebug>
#include <QFileDialog>
#include <QSound>

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
	lastFocusScores(1000, 0)
{
	this->ui->setupUi(this);

	this->ui->chkGuardarVideo->setChecked(!::IRIS_VIDEO_CAPTURE.isPaused());
	QObject::connect(this->ui->chkGuardarVideo, SIGNAL(stateChanged(int)), &::IRIS_VIDEO_CAPTURE, SLOT(setPause(int)));

	qRegisterMetaType<VideoProcessor>("VideoProcessor");
	QObject::connect(&::PROCESSING_THREAD, SIGNAL(signalFrameProcessed(VideoProcessor)), this->ui->video, SLOT(slotFrameProcessed(VideoProcessor)), Qt::BlockingQueuedConnection);
}

MainWindow::~MainWindow()
{
	QObject::disconnect(&::PROCESSING_THREAD, SIGNAL(signalFrameProcessed(VideoProcessor)), this->ui->video, SLOT(slotFrameProcessed(VideoProcessor)));
    delete ui;
}

void MainWindow::slotFrameAvailable(const ColorImage& /*frame*/)
{
}

void MainWindow::slotFrameProcessed(const VideoProcessor& videoProcessor)
{
	VideoProcessor::VideoStatus status = videoProcessor.lastStatus;

	mostrarEnfoque(videoProcessor.lastFocusScore, videoProcessor.parameters.focusThreshold, videoProcessor.lastFrame.cols);

	int irisScore = ((status >= VideoProcessor::FOCUSED_NO_IRIS) ? videoProcessor.lastIrisQuality : 0);
	this->ui->irisScore->setValue(irisScore);
}

void MainWindow::slotGotTemplate(const VideoProcessor& videoProcessor)
{
	//this->lastTemplate = videoProcessor.getTemplate();
	this->lastTemplate = videoProcessor.getAverageTemplate();
	this->lastIrisFrameSegmentation = videoProcessor.getBestTemplateSegmentation();
	this->lastIrisFrame = videoProcessor.getBestTemplateFrame().clone();

	cvtColor(this->lastIrisFrame, this->decoratedFrame, CV_GRAY2RGB);

	this->decorator.setDrawingColors();
	this->decorator.lineWidth = 1;
	this->decorator.drawSegmentationResult(this->decoratedFrame, this->lastIrisFrameSegmentation);

	// Muestra el frame decorado en la ventana de la imagen capturada. Para esto, hay que resizearlo
	cv::resize(this->decoratedFrame, this->resizedFrame, Size(320*1.2, 240*1.2));
	this->decorator.drawTemplate(this->resizedFrame, this->lastTemplate);

	Mat region = this->resizedFrame(Range(this->resizedFrame.rows-30, this->resizedFrame.rows), Range::all());
	decorator.drawIrisTexture(this->lastIrisFrame, region, this->lastIrisFrameSegmentation);

	this->ui->capturedImage->showImage(this->resizedFrame);

	if (this->ui->autoIdentify->isChecked()) {
		this->identifyTemplate(this->lastTemplate, this->lastIrisFrame, this->lastIrisFrameSegmentation);
	}
}

void MainWindow::on_btnIdentificar_clicked()
{
	if (this->lastIrisFrame.empty()) return;				// Sólo hacer identificación si hay un iris

	this->identifyTemplate(this->lastTemplate, this->lastIrisFrame, this->lastIrisFrameSegmentation);
}

void MainWindow::on_btnRegistrar_clicked()
{
	if (this->lastIrisFrame.empty()) return;				// Sólo registrar si hay un iris

	this->registerTemplate(this->lastTemplate, this->lastIrisFrame, this->lastIrisFrameSegmentation);
}

void MainWindow::on_btnGuardarImagen_clicked()
{
	ofrecerGuardarImagen(this->lastIrisFrame);
}

void MainWindow::on_btnCapturar_clicked()
{
	//ofrecerGuardarImagen(this->lastFrame);
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
	//Mat imagen = this->lastFrame.clone();

	// Genera un template a partir de la imagen
	Segmentator segmentator;

}

void MainWindow::identifyTemplate(const IrisTemplate& irisTemplate, const GrayscaleImage& image, const SegmentationResult& segmentationResult)
{
	if (this->matchingDialog.isVisible()) return;

	this->matchingDialog.doMatch(irisTemplate, image, segmentationResult);
}

void MainWindow::registerTemplate(const IrisTemplate& irisTemplate, const GrayscaleImage& image, const SegmentationResult& segmentationResult)
{
	if (this->registerDialog.isVisible()) return;

	this->registerDialog.doRegister(irisTemplate, image, segmentationResult);
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
	/*if (this->lastFrame.empty()) return;

	cvtColor(this->lastFrame, this->lastIrisFrame, CV_BGR2GRAY);

	this->lastIrisFrameSegmentation = ::PROCESSING_THREAD.videoProcessor.segmentator.segmentImage(this->lastIrisFrame);
	this->lastTemplate = ::PROCESSING_THREAD.videoProcessor.irisEncoder.generateTemplate(this->lastIrisFrame, this->lastIrisFrameSegmentation);
	this->identifyTemplate(this->lastTemplate, this->lastIrisFrame, this->lastIrisFrameSegmentation);*/
}
