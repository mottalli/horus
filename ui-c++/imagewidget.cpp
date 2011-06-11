#include <QPainter>

#include "imagewidget.h"

ImageWidget::ImageWidget(QWidget *parent) :
    QWidget(parent)
{
}

void ImageWidget::showImage(const Image& image)
{
	switch (image.type()) {
	case CV_8UC1:
		cvtColor(image, _tmp, CV_GRAY2RGB);
		break;
	case CV_8UC3:
		cvtColor(image, _tmp, CV_BGR2RGB);
		break;
	}

	assert(_tmp.isContinuous());
	_qimage = QImage(_tmp.data, _tmp.cols, _tmp.rows, QImage::Format_RGB888);

	this->setFixedSize(image.cols, image.rows);

	repaint();
}

void ImageWidget::paintEvent(QPaintEvent* /*event*/)
{
	QPainter painter(this);
	painter.drawImage(QPoint(0,0), _qimage);
	painter.end();
}
