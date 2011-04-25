#include <QDebug>
#include <QFileDialog>

#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
	ui(new Ui::MainWindow),
	lastFocusScores(100, 0)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::slotFrameAvailable(const Mat& /*frame*/)
{
}

void MainWindow::slotFrameProcessed(const VideoProcessor& videoProcessor)
{
	VideoProcessor::VideoStatus status = videoProcessor.lastStatus;

	this->lastFocusScores.pop_front();
	this->lastFocusScores.push_back(videoProcessor.lastFocusScore);


	videoProcessor.lastFrame.copyTo(this->lastFrame);

	int heightControl = 50;
	decorator.drawFocusScores(this->lastFocusScores, this->lastFrame, Rect(0, this->lastFrame.rows-heightControl, this->lastFrame.cols, heightControl), videoProcessor.parameters.focusThreshold);


	switch (status) {

	case VideoProcessor::IRIS_TOO_CLOSE:
	case VideoProcessor::IRIS_TOO_FAR:
	case VideoProcessor::INTERLACED:
	case VideoProcessor::IRIS_LOW_QUALITY:
	case VideoProcessor::GOT_TEMPLATE:
		ui->focusScore->setValue(videoProcessor.lastFocusScore);
	case VideoProcessor::UNPROCESSED:
	case VideoProcessor::DEFOCUSED:
		ui->video->showImage(this->lastFrame);
		break;

	case VideoProcessor::FOCUSED_NO_IRIS:
	case VideoProcessor::FOCUSED_IRIS:
		Mat tmp = this->lastFrame.clone();
		decorator.drawSegmentationResult(tmp, videoProcessor.lastSegmentationResult);
		ui->video->showImage(tmp);
		break;
	}
}

void MainWindow::slotGotTemplate(const VideoProcessor& videoProcessor)
{
	qDebug() << "Template!";

	this->lastTemplate = const_cast<VideoProcessor&>(videoProcessor).getTemplate();
	this->lastIrisFrameSegmentation = videoProcessor.getTemplateSegmentation();
	videoProcessor.getTemplateFrame().copyTo(this->lastIrisFrame);

	this->lastIrisFrame.convertTo(this->decoratedFrame, CV_GRAY2BGR);

	this->decorator.drawSegmentationResult(this->decoratedFrame, this->lastIrisFrameSegmentation);

	// Muestra el frame decorado en la ventana de la imagen capturada. Para esto, hay que resizearlo
	cv::resize(this->decoratedFrame, this->resizedFrame, Size(), 0.5, 0.5);
	this->decorator.drawTemplate(this->resizedFrame, this->lastTemplate);
	this->ui->capturedImage->showImage(this->resizedFrame);
}

void MainWindow::on_btnIdentificar_clicked()
{
}

void MainWindow::on_btnRegistrar_clicked()
{

}

void MainWindow::on_btnGuardarImagen_clicked()
{
	ofrecerGuardarImagen(this->lastIrisFrame);
}

void MainWindow::on_btnCapturar_clicked()
{
	ofrecerGuardarImagen(this->lastFrame);
}

void MainWindow::ofrecerGuardarImagen(const Mat& imagen)
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
	Mat imagen = this->lastFrame.clone();

	// Genera un template a partir de la imagen
	Segmentator segmentator;

}
