#ifndef IRISVIDEOWIDGET_H
#define IRISVIDEOWIDGET_H

#include "imagewidget.h"

class IrisVideoWidget : public ImageWidget
{
    Q_OBJECT
public:
    explicit IrisVideoWidget(QWidget *parent = 0);

signals:

public slots:
	void slotFrameProcessed(const VideoProcessor& videoProcessor);

private:
	static void drawCrosshair(Image& image, Point p, int thickness = 1, int size=25, Scalar color=CV_RGB(255,255,255));
	ColorImage decoratedFrame;
	Decorator decorator;

};

#endif // IRISVIDEOWIDGET_H
