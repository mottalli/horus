#ifndef IMAGEWIDGET_H
#define IMAGEWIDGET_H

#include <QWidget>
#include <QImage>

#include "common.h"

class ImageWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ImageWidget(QWidget *parent = 0);

	QSize sizeHint() const { return _qimage.size(); }
	QSize minimumSizeHint() const { return _qimage.size(); }

signals:

public slots:
	void showImage(const Mat& image);
protected:
	void paintEvent(QPaintEvent* event);
	QImage _qimage;
	Mat _tmp;
};

#endif // IMAGEWIDGET_H
