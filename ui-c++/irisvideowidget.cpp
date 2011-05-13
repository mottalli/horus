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
		this->decorator.drawSegmentationResult(this->decoratedFrame, videoProcessor.lastSegmentationResult);

		this->drawCrosshair(this->decoratedFrame, Point(videoProcessor.lastSegmentationResult.irisCircle.xc, videoProcessor.lastSegmentationResult.irisCircle.yc), 1);

		rectangle(this->decoratedFrame, videoProcessor.eyeROI, CV_RGB(255,255,255));

		if (status >= VideoProcessor::FOCUSED_IRIS) {
			// Efecto ciencia ficción!
			double q = min<double>(1.0, double(videoProcessor.templateBuffer.size()) / double(videoProcessor.parameters.minCountForTemplateAveraging));
			double angle = q*2*M_PI;
			int width = int(400.0*q);
			int height = (videoProcessor.lastSegmentationResult.irisCircle.radius-videoProcessor.lastSegmentationResult.pupilCircle.radius)/2;
			vector< pair<Point, Point> > pts = Tools::iterateIris(videoProcessor.lastSegmentationResult, width, height, -M_PI/2, angle-M_PI/2);
			for (size_t i = 0; i < pts.size(); i++) {
				Point p = pts[i].second;
				Vec3f val = this->decoratedFrame.at<Vec3b>(p);
				Vec3f color;
				double alpha;
				if (status == VideoProcessor::FINISHED_CAPTURE) {
					//color =  Vec3f(0,0,255);
					color =  Vec3f(0,128,0);
					alpha = 0.2;
				} else {
					alpha = 0.8;
					color = Vec3f(0,255,255);
				}
				Vec3b final = Vec3f( val[0]*alpha+color[0]*(1.0-alpha), val[1]*alpha+color[1]*(1.0-alpha), val[2]*alpha+color[2]*(1.0-alpha) );

				this->decoratedFrame.at<Vec3b>(p) = final;
			}
		}

		/*if (status == VideoProcessor::FINISHED_CAPTURE) {
			if (boost::filesystem::is_regular_file("./Ding.wav")) {
				QSound::play("./Ding.wav");
			} else {
				//this->decoratedFrame.setTo(Scalar(255,255,255));
			}
		}*/
	}

	this->drawCrosshair(this->decoratedFrame, Point(this->decoratedFrame.cols/2, this->decoratedFrame.rows/2), 2);
	this->showImage(this->decoratedFrame);
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
