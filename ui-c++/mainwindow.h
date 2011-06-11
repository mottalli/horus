#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "common.h"
#include "matchingdialog.h"
#include "registerdialog.h"
#include "debugdialog.h"

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

	void ofrecerGuardarImagen(const Image& imagen);

public slots:
	void slotFrameAvailable(const ColorImage& frame);
	void slotFrameProcessed(const VideoProcessor& videoProcessor);
	void slotGotTemplate(const VideoProcessor& videoProcessor);

private slots:
	void on_btnIdentificar_clicked();
	void on_btnRegistrar_clicked();
	void on_btnGuardarImagen_clicked();
	void on_btnCapturar_clicked();
	void on_btnForzarRegistracion_clicked();

	void on_btnForzarIdentificacion_clicked();

	void on_debugWindow_clicked();

private:
	void identifyTemplate(const IrisTemplate& irisTemplate, const GrayscaleImage& image, const SegmentationResult& segmentationResult);
	void registerTemplate(const IrisTemplate& irisTemplate, const GrayscaleImage& image, const SegmentationResult& segmentationResult);
	void mostrarEnfoque(double enfoque, double threshold, int width);
	void showTemplateImage();

	static QString statusToString(VideoProcessor::VideoStatus status);

	Image lastProcessedFrame;

    Ui::MainWindow *ui;
	Decorator decorator;

	ColorImage decoratedFrame, resizedFrame;
	GrayscaleImage lastIrisFrame;
	ColorImage imagenEnfoque;
	SegmentationResult lastIrisFrameSegmentation;
	IrisTemplate lastTemplate;
	list<double> lastFocusScores;

	MatchingDialog matchingDialog;
	RegisterDialog registerDialog;
	DebugDialog debugDialog;
};

#endif // MAINWINDOW_H
