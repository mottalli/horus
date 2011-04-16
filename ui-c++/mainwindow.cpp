#include <QDebug>

#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
	ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::slotFrameAvailable(const Mat& frame)
{
	//ui->video->showImage(frame);
}

void MainWindow::slotFrameProcessed(const VideoProcessor& videoProcessor)
{
	VideoProcessor::VideoStatus status = videoProcessor.lastStatus;

	switch (status) {

	case VideoProcessor::IRIS_TOO_CLOSE:
	case VideoProcessor::IRIS_TOO_FAR:
	case VideoProcessor::INTERLACED:
	case VideoProcessor::IRIS_LOW_QUALITY:
	case VideoProcessor::GOT_TEMPLATE:
		ui->focusScore->setValue(videoProcessor.lastFocusScore);
	case VideoProcessor::UNPROCESSED:
	case VideoProcessor::DEFOCUSED:
		ui->video->showImage(videoProcessor.lastFrame);
		break;
	case VideoProcessor::FOCUSED_NO_IRIS:
	case VideoProcessor::FOCUSED_IRIS:
		Mat tmp = videoProcessor.lastFrame.clone();
		decorator.drawSegmentationResult(tmp, videoProcessor.lastSegmentationResult);
		ui->video->showImage(tmp);
		break;
	}
}

void MainWindow::slotGotTemplate(const VideoProcessor& videoProcessor)
{
	IrisTemplate irisTemplate = const_cast<VideoProcessor&>(videoProcessor).getTemplate();
	qDebug() << "Template!";
}
