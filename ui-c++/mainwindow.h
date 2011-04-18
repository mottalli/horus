#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <horus/videoprocessor.h>
#include <horus/decorator.h>

#include "common.h"


namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
	void slotFrameAvailable(const Mat& frame);
	void slotFrameProcessed(const VideoProcessor& videoProcessor);
	void slotGotTemplate(const VideoProcessor& videoProcessor);

private:
    Ui::MainWindow *ui;
	Decorator decorator;
};

#endif // MAINWINDOW_H
