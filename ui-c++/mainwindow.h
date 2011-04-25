#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "common.h"
#include "matchingdialog.h"


namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

	void ofrecerGuardarImagen(const Mat& imagen);

public slots:
	void slotFrameAvailable(const Mat& frame);
	void slotFrameProcessed(const VideoProcessor& videoProcessor);
	void slotGotTemplate(const VideoProcessor& videoProcessor);

private slots:
	void on_btnIdentificar_clicked();
	void on_btnRegistrar_clicked();
	void on_btnGuardarImagen_clicked();
	void on_btnCapturar_clicked();
	void on_btnForzarRegistracion_clicked();

private:
	void identificarTemplate(const IrisTemplate& irisTemplate, const Mat_<uint8_t>& imagen, const SegmentationResult& segmentationResult);
	void mostrarEnfoque(double enfoque, double threshold, int width);

    Ui::MainWindow *ui;
	Decorator decorator;

	Mat lastFrame, decoratedFrame, resizedFrame;
	Mat_<uint8_t> lastIrisFrame;
	Mat imagenEnfoque;
	SegmentationResult lastIrisFrameSegmentation;
	IrisTemplate lastTemplate;
	list<double> lastFocusScores;

	MatchingDialog matchingDialog;
};

#endif // MAINWINDOW_H
