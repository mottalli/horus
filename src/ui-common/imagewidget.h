#pragma once
#include <qt5/QtWidgets/QWidget>
#include <qt5/QtGui/QImage>

#include <opencv2/opencv.hpp>

class ImageWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ImageWidget(QWidget *parent = 0);

    QSize sizeHint() const { return _qimage.size(); }
    QSize minimumSizeHint() const { return _qimage.size(); }

signals:

public slots:
    void showImage(const cv::Mat& image);

protected:
    void paintEvent(QPaintEvent* event);
    QImage _qimage;
    cv::Mat _tmp;
};
