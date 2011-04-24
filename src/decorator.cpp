/*
 * decorator.cpp
 *
 *  Created on: Jun 4, 2009
 *      Author: marcelo
 */

#include "decorator.h"
#include "irisencoder.h"
#include "tools.h"
#include <cmath>

Decorator::Decorator()
{
	this->irisColor = Scalar(0,255,0);
	this->pupilColor = Scalar(255,0,0);
	this->upperEyelidColor = Scalar(0,0,255);
	this->lowerEyelidColor = Scalar(0,0,255);
}

void Decorator::drawSegmentationResult(Mat& image, const SegmentationResult& segmentationResult) const
{
	Scalar irisColor_ = (image.channels() == 1 ? Scalar(255,255,255) : this->irisColor);
	Scalar pupilColor_ = (image.channels() == 1 ? Scalar(255,255,255) : this->pupilColor);
	this->drawContour(image, segmentationResult.irisContour, irisColor_);
	this->drawContour(image, segmentationResult.pupilContour, pupilColor_);

	const Circle& irisCircle = segmentationResult.irisCircle;

	if (segmentationResult.eyelidsSegmented) {
		int xMin = segmentationResult.irisCircle.xc-segmentationResult.irisCircle.radius;
		int xMax = segmentationResult.irisCircle.xc+segmentationResult.irisCircle.radius;
		this->drawParabola(image, segmentationResult.upperEyelid, xMin, xMax, this->upperEyelidColor);
		this->drawParabola(image, segmentationResult.lowerEyelid, xMin, xMax, this->lowerEyelidColor);
	}
}

void Decorator::drawEncodingZone(Mat& image, const SegmentationResult& segmentationResult)
{
	bool fill = false;
	int step = 5, stepCounter = step;

	const int width = 512, height = 80;

	std::vector< std::pair<Point, Point> > irisPoints = Tools::iterateIris(segmentationResult,
		width, height, IrisEncoder::THETA0,
		IrisEncoder::THETA1, IrisEncoder::RADIUS_TO_USE);

	for (size_t i = 0; i < irisPoints.size(); i++) {
		Point imagePoint = irisPoints[i].second;
		Point coord = irisPoints[i].first;
		int x = int(imagePoint.x), y = int(imagePoint.y);

		if (x < 0 || x >= image.size().width  || y < 0 || y > image.size().height) {
			continue;
		}

		if (fill || (coord.x == 0 || coord.x == width-1 || coord.y == 0 || coord.y == height-1)) {
			if ( (stepCounter--) > 0) {
				continue;
			}
			
			if (image.channels() == 1) {
				image.at<uchar>(y,x) = 255;
			} else if (image.channels() == 3) {
				image.at<Vec3b>(y, x) = Vec3b(0,255,255);
			}
			stepCounter=step;
		}
	}

}

void Decorator::drawContour(Mat& image, const Contour& contour, const Scalar& color) const
{
	if (contour.size() < 2) return;

	const Point p0 = contour[0];
	int n = contour.size();

	Point lastPoint = p0;

	Mat_<Vec3b>& image3 = (Mat_<Vec3b>&)image;

	for (int i = 1; i < n; i++) {
		const Point p = contour[i];
		//image.at<Vec3b>(p.y, p.x) = Vec3b(0, 255, 255);
		line(image, lastPoint, p, color, 1);
		lastPoint = p;
	}

	line(image, lastPoint, p0, color, 1);
}

void Decorator::drawParabola(Mat& image, const Parabola& parabola, int xMin, int xMax, const Scalar& color) const
{
	if (xMin < 0) xMin = 1;
	if (xMax < 0 || xMax >= image.cols) xMax = image.cols-1;

	Point lastPoint = Point(xMin, int(parabola.value(xMin)));
	for (int x = xMin+1; x <= xMax; x++) {
		Point point = Point(x, int(parabola.value(x)));

		line(image, lastPoint, point, color, 1);
		lastPoint = point;
	}
}

void Decorator::drawTemplate(Mat& image, const IrisTemplate& irisTemplate)
{
	Mat imgTemplate = irisTemplate.getTemplateImage();
	Mat mask = irisTemplate.getUnpackedMask();

	mask.setTo(Scalar(255), mask);
	bitwise_not(mask, mask);			// Hacky way to NOT the template
	imgTemplate.setTo(Scalar(127), mask);

	Mat decoratedTemplate;

	if (imgTemplate.rows < 10) {
		Mat tmp(3*imgTemplate.rows+2, imgTemplate.cols+2, CV_8U, Scalar(128));

		int width = imgTemplate.cols;
		for (int i = 0; i < imgTemplate.rows; i++) {
			Mat r = tmp(Rect(1, 3*i+2, imgTemplate.cols, 1));
			imgTemplate.row(i).copyTo(r);
		}
		resize(tmp, decoratedTemplate, Size(1.5*tmp.cols, 2.5*tmp.rows), 1, 1, CV_INTER_NN);
	} else {
		decoratedTemplate = imgTemplate.clone();
	}

	Rect templateRect(15, 15, decoratedTemplate.cols, decoratedTemplate.rows);

	if (image.channels() == 3) {
		vector<Mat> channels(3, decoratedTemplate);
		Mat part = image(templateRect);
		merge(channels, part);
	} else {
		Mat m = image(templateRect);
		decoratedTemplate.copyTo(m);
	}
	Point p0(templateRect.x-1, templateRect.y-1);
	Point p1(p0.x + templateRect.width+1, p0.y+templateRect.height+1);
	rectangle(image, p0, p1, CV_RGB(0,0,0), 1);
}

void Decorator::setDrawingColors(Scalar pupilColor_, Scalar irisColor_, Scalar upperEyelidColor_, Scalar lowerEyelidColor_)
{
	this->pupilColor = pupilColor_;
	this->irisColor = irisColor_;
	this->upperEyelidColor = upperEyelidColor_;
	this->lowerEyelidColor = lowerEyelidColor_;
}

void Decorator::drawFocusScores(const list<double>& focusScores, Mat image, Rect rect, double threshold)
{
	image(rect) = Scalar(255,255,255);
	rectangle(image, rect, Scalar(0,0,0), 1);

	if (threshold > 0) {
		double y = rect.height - ((threshold/100.0) * rect.height);
		double yimg = rect.y + y;
		line(image, Point(rect.x, yimg), Point(rect.x+rect.width, yimg), Scalar(255,0,0));
	}

	Point lastPoint;

	//for (int i = focusScores.size()-1; i >= 0 && (rect.width - (focusScores.size()-1-i)) > 0; i--) {
	int i =  i = focusScores.size()-1;
	for (list<double>::const_reverse_iterator it = focusScores.rbegin(); it != focusScores.rend(); it++) {
		double focusScore = *it;
		double x = rect.width - (focusScores.size()-1-i);

		if (x <= 0) break;

		double y = rect.height - ((focusScore/100.0) * rect.height);

		Point pimg = rect.tl() + Point(x,y);
		if (lastPoint.x == 0 && lastPoint.y == 0) {
			lastPoint = pimg;
		}

		line(image, lastPoint, pimg, Scalar(0,0,255));
		lastPoint = pimg;
		i--;
	}
}
