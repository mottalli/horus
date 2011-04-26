#include <QDebug>
#include <QFileDialog>

#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
	ui(new Ui::MainWindow),
	lastFocusScores(1000, 0)
{
	this->ui->setupUi(this);
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
	videoProcessor.lastFrame.copyTo(this->lastFrame);

	mostrarEnfoque(videoProcessor.lastFocusScore, videoProcessor.parameters.focusThreshold, this->lastFrame.cols);

	Mat frameDecorado = this->lastFrame;

	if (status >= VideoProcessor::FOCUSED_NO_IRIS) {
		Scalar pupilColor = (status == VideoProcessor::FOCUSED_NO_IRIS ? (Scalar)CV_RGB(255,255,255) :  Decorator::DEFAULT_PUPIL_COLOR);
		Scalar irisColor = (status == VideoProcessor::FOCUSED_NO_IRIS ? (Scalar)CV_RGB(255,255,255) : Decorator::DEFAULT_IRIS_COLOR);
		decorator.lineWidth = (status == VideoProcessor::FOCUSED_NO_IRIS ? 1 : 2);

		frameDecorado = this->lastFrame.clone();
		decorator.setDrawingColors(pupilColor, irisColor);
		decorator.drawSegmentationResult(frameDecorado, videoProcessor.lastSegmentationResult);

		this->drawCrosshair(frameDecorado, Point(videoProcessor.lastSegmentationResult.irisCircle.xc, videoProcessor.lastSegmentationResult.irisCircle.yc), 1);
	}

	this->drawCrosshair(frameDecorado, Point(frameDecorado.cols/2, frameDecorado.rows/2), 2);
	this->ui->video->showImage(frameDecorado);
}

void MainWindow::slotGotTemplate(const VideoProcessor& videoProcessor)
{
	qDebug() << "Template!";

	//this->lastTemplate = videoProcessor.getTemplate();
	this->lastTemplate = videoProcessor.getAverageTemplate();
	this->lastIrisFrameSegmentation = videoProcessor.getBestTemplateSegmentation();
	//videoProcessor.getBestTemplateFrame().copyTo(this->lastIrisFrame);
	this->lastIrisFrame = videoProcessor.getBestTemplateFrame().clone();

	cvtColor(this->lastIrisFrame, this->decoratedFrame, CV_GRAY2RGB);

	this->decorator.setDrawingColors();
	this->decorator.drawSegmentationResult(this->decoratedFrame, this->lastIrisFrameSegmentation);

	// Muestra el frame decorado en la ventana de la imagen capturada. Para esto, hay que resizearlo
	cv::resize(this->decoratedFrame, this->resizedFrame, Size(), 0.5, 0.5);
	this->decorator.drawTemplate(this->resizedFrame, this->lastTemplate);
	this->ui->capturedImage->showImage(this->resizedFrame);
}

void MainWindow::on_btnIdentificar_clicked()
{
	if (this->lastIrisFrame.empty()) return;				// Sólo hacer identificación si hay un iris

	this->identificarTemplate(this->lastTemplate, this->lastIrisFrame, this->lastIrisFrameSegmentation);
}

void MainWindow::on_btnRegistrar_clicked()
{
	if (this->lastIrisFrame.empty()) return;				// Sólo registrar si hay un iris
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

void MainWindow::identificarTemplate(const IrisTemplate& irisTemplate, const Mat_<uint8_t>& imagen, const SegmentationResult& segmentationResult)
{
	if (this->matchingDialog.isVisible()) return;

	this->matchingDialog.doMatch(irisTemplate, imagen, segmentationResult);
}

void MainWindow::mostrarEnfoque(double enfoque, double threshold, int width)
{
	this->lastFocusScores.pop_front();
	this->lastFocusScores.push_back(enfoque);

	if (this->imagenEnfoque.empty()) {
		this->imagenEnfoque.create(Size(width, 30), CV_8UC3);
	}

	decorator.drawFocusScores(this->lastFocusScores, this->imagenEnfoque, Rect(0, 0, this->imagenEnfoque.cols, this->imagenEnfoque.rows), threshold);
	this->ui->animacionEnfoque->showImage(this->imagenEnfoque);
	this->ui->focusScore->setValue(enfoque);
}

void MainWindow::drawCrosshair(Mat& image, Point p, int thickness, int size, Scalar color)
{
	Point p00 = Point(p.x, p.y-size/2);
	Point p01 = Point(p.x, p.y+size/2);
	Point p10 = Point(p.x-size/2, p.y);
	Point p11 = Point(p.x+size/2, p.y);

	line(image, p00, p01, color, thickness);
	line(image, p10, p11, color, thickness);
}
