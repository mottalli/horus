#include "irisvideowidget.h"

IrisVideoWidget::IrisVideoWidget(QWidget *parent) :
    ImageWidget(parent)
{
}

void IrisVideoWidget::slotFrameProcessed(const VideoProcessor& videoProcessor)
{
    VideoProcessor::VideoStatus status = videoProcessor.lastStatus;

    videoProcessor.lastFrame.copyTo(this->decoratedFrame);
    if (status >= VideoProcessor::FOCUSED_NO_IRIS) {
        Scalar pupilColor = (status >= VideoProcessor::FOCUSED_IRIS ? Decorator::DEFAULT_PUPIL_COLOR : (Scalar)CV_RGB(255,255,255));
        Scalar irisColor = (status >= VideoProcessor::FOCUSED_IRIS ? Decorator::DEFAULT_IRIS_COLOR : (Scalar)CV_RGB(255,255,255));
        this->decorator.lineWidth = (status >= VideoProcessor::FOCUSED_IRIS ? 2 : 1);

        this->decorator.setDrawingColors(pupilColor, irisColor);

        this->decorator.drawCaptureStatus(this->decoratedFrame, videoProcessor);
        this->decorator.drawSegmentationResult(this->decoratedFrame, videoProcessor.lastSegmentationResult);


        if (status >= VideoProcessor::FOCUSED_IRIS) {
            //decorator.drawTemplate(this->decoratedFrame, videoProcessor.lastTemplate);
        }
    }

    this->drawCrosshair(this->decoratedFrame, Point(this->decoratedFrame.cols/2, this->decoratedFrame.rows/2), 2);

    if (this->decoratedFrame.cols <= 800) {
        this->previewImage = this->decoratedFrame;
    } else {
        //this->decoratedFrame.resize();
        float factor = 800.0f/this->decoratedFrame.cols;
        cv::resize(this->decoratedFrame, this->previewImage, Size(), factor, factor);
    }
    this->showImage(this->previewImage);
}


void IrisVideoWidget::drawCrosshair(Image& image, Point p, int thickness, int size, Scalar color)
{
    Point p00 = Point(p.x, p.y-size/2);
    Point p01 = Point(p.x, p.y+size/2);
    Point p10 = Point(p.x-size/2, p.y);
    Point p11 = Point(p.x+size/2, p.y);

    line(image, p00, p01, color, thickness);
    line(image, p10, p11, color, thickness);
}
