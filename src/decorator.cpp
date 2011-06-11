/*
 * decorator.cpp
 *
 *  Created on: Jun 4, 2009
 *      Author: marcelo
 */
#include <cmath>

#include "decorator.h"
#include "irisencoder.h"
#include "tools.h"

using namespace horus;

const Scalar Decorator::DEFAULT_PUPIL_COLOR = CV_RGB(255,0,0);
const Scalar Decorator::DEFAULT_IRIS_COLOR = CV_RGB(0,255,0);
const Scalar Decorator::DEFAULT_EYELID_COLOR = CV_RGB(0,0,255);

Decorator::Decorator()
{
	this->irisColor = Decorator::DEFAULT_IRIS_COLOR;
	this->pupilColor = Decorator::DEFAULT_PUPIL_COLOR;
	this->upperEyelidColor = Decorator::DEFAULT_EYELID_COLOR;
	this->lowerEyelidColor = Decorator::DEFAULT_EYELID_COLOR;
	this->lineWidth = 1;
}

void Decorator::drawSegmentationResult(Image& image, const SegmentationResult& segmentationResult) const
{
	Scalar irisColor_ = (image.channels() == 1 ? (Scalar)CV_RGB(255,255,255) : this->irisColor);
	Scalar pupilColor_ = (image.channels() == 1 ? (Scalar)CV_RGB(255,255,255) : this->pupilColor);
	this->drawContour(image, segmentationResult.irisContour, irisColor_);
	this->drawContour(image, segmentationResult.pupilContour, pupilColor_);

	if (segmentationResult.eyelidsSegmented) {
		int xMin = segmentationResult.irisCircle.center.x-segmentationResult.irisCircle.radius;
		int xMax = segmentationResult.irisCircle.center.x+segmentationResult.irisCircle.radius;
		this->drawParabola(image, segmentationResult.upperEyelid, xMin, xMax, this->upperEyelidColor);
		this->drawParabola(image, segmentationResult.lowerEyelid, xMin, xMax, this->lowerEyelidColor);
	}
}

void Decorator::drawEncodingZone(Image& image, const SegmentationResult& segmentationResult)
{
	const int width = 512, height = 80;

	vector< pair<Point, Point> > irisPoints = tools::iterateIris(segmentationResult,
		width, height, IrisEncoder::THETA0,
		IrisEncoder::THETA1, IrisEncoder::MIN_RADIUS_TO_USE, IrisEncoder::MAX_RADIUS_TO_USE);

	Contour outerContour(width), innerContour(width);
	pair<Point, Point> firstBorder, secondBorder;

	for (size_t i = 0; i < irisPoints.size(); i++) {
		Point imagePoint = irisPoints[i].second;
		Point coord = irisPoints[i].first;
		int x = imagePoint.x, y = imagePoint.y;

		if (x < 0 || x >= image.size().width  || y < 0 || y > image.size().height) {
			continue;
		}

		if (coord.y == 0) {
			innerContour[coord.x] = imagePoint;
			if (coord.x == 0) {
				firstBorder.first = imagePoint;
			} else if (coord.x == width-1) {
				secondBorder.first = imagePoint;
			}
		} else if (coord.y == height-1) {
			outerContour[coord.x] = imagePoint;
			if (coord.x == 0) {
				firstBorder.second= imagePoint;
			} else if (coord.x == width-1) {
				secondBorder.second = imagePoint;
			}
		}

		//TODO: fill
	}

	line(image, firstBorder.first, firstBorder.second, CV_RGB(255,255,0), this->lineWidth);
	line(image, secondBorder.first, secondBorder.second, CV_RGB(255,255,0), this->lineWidth);

	assert(innerContour.size() == outerContour.size());
	Point ip0 = innerContour[0], op0 = outerContour[0];
	Point ip1, op1;

	for (size_t i = 1; i < innerContour.size(); i++) {
		ip1 = innerContour[i];
		op1 = outerContour[i];
		line(image, ip0, ip1, CV_RGB(255,255,0), this->lineWidth);
		line(image, op0, op1, CV_RGB(255,255,0), this->lineWidth);

		ip0 = ip1;
		op0 = op1;
	}

}

void Decorator::drawContour(Image& image, const Contour& contour, const Scalar& color) const
{
	if (contour.size() < 2) return;

	const Point p0 = contour[0];
	int n = contour.size();

	Point lastPoint = p0;

	for (int i = 1; i < n; i++) {
		const Point p = contour[i];
		line(image, lastPoint, p, color, this->lineWidth);
		lastPoint = p;
	}

	line(image, lastPoint, p0, color, this->lineWidth);
}

void Decorator::drawParabola(Image& image, const Parabola& parabola, int xMin, int xMax, const Scalar& color) const
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

void Decorator::drawTemplate(Image& image, const IrisTemplate& irisTemplate, Point p0)
{
	GrayscaleImage imgTemplate = irisTemplate.getTemplateImage();
	Decorator::superimposeImage(imgTemplate, image, p0, true);
}

void Decorator::setDrawingColors(Scalar pupilColor_, Scalar irisColor_, Scalar upperEyelidColor_, Scalar lowerEyelidColor_)
{
	this->pupilColor = pupilColor_;
	this->irisColor = irisColor_;
	this->upperEyelidColor = upperEyelidColor_;
	this->lowerEyelidColor = lowerEyelidColor_;
}

void Decorator::drawFocusScores(Mat& image, const list<double>& focusScores, Rect rect, double threshold)
{
	image(rect) = CV_RGB(255,255,255);

	if (threshold > 0) {
		double y = rect.height - ((threshold/100.0) * rect.height);
		double yimg = rect.y + y;
		line(image, Point(rect.x, yimg), Point(rect.x+rect.width, yimg), CV_RGB(255,0,0));
	}

	Point lastPoint;

	int i = focusScores.size()-1;
	for (list<double>::const_reverse_iterator it = focusScores.rbegin(); it != focusScores.rend(); it++) {
		double focusScore = *it;
		double x = rect.width - (focusScores.size()-1-i);

		if (x <= 0) break;

		double y = rect.height - ((focusScore/100.0) * rect.height);

		Point pimg = rect.tl() + Point(x,y);
		if (lastPoint.x == 0 && lastPoint.y == 0) {
			lastPoint = pimg;
		}

		line(image, lastPoint, pimg, CV_RGB(0,0,255));
		lastPoint = pimg;
		i--;
	}
}

void Decorator::drawIrisTexture(const Mat& imageSrc, Mat& imageDest, SegmentationResult segmentationResult)
{
	GrayscaleImage texture(imageDest.size()), mask(imageDest.size());
	GrayscaleImage srcBW;

	tools::toGrayscale(imageSrc, srcBW, false);

	IrisEncoder::normalizeIris(srcBW, texture, mask, segmentationResult);

	tools::stretchHistogram(texture, texture);
	Decorator::superimposeImage(texture, imageDest, Point(0,0), false);
}

void Decorator::superimposeImage(const Image& imageSrc, Image& imageDest, Point p, bool drawBorder)
{
	Rect r = Rect(p.x, p.y, imageSrc.cols, imageSrc.rows);
	assert(r.br().x < imageDest.cols && r.br().y < imageDest.rows);		// Inside the image

	Image destRect = imageDest(r);

	if (imageSrc.type() == imageDest.type()) {
		imageSrc.copyTo(destRect);
	} else if (imageSrc.channels() == 3) {
		assert(imageDest.channels() == 1);
		vector<Mat> channels(3, imageSrc);
		merge(channels, destRect);
	} else {
		assert(imageSrc.channels() == 1 && imageDest.channels() == 3);
		cvtColor(imageSrc, destRect, CV_GRAY2BGR);
	}

	if (drawBorder) {
		Point tl(r.tl().x-1, r.tl().y-1);
		Point br(r.br().x+1, r.br().y+1);
		rectangle(imageDest, tl, br, CV_RGB(0,0,0), 1);
	}
}

void Decorator::drawCaptureStatus(Image& image, const VideoProcessor& videoProcessor)
{
	//TODO: Support for B&W images
	assert(image.channels() == 3);

	VideoProcessor::VideoStatus status = videoProcessor.lastStatus;

	if (status >= VideoProcessor::FOCUSED_IRIS) {
		// Science fiction effect!
		double q = min<double>(1.0, double(videoProcessor.templateBuffer.size()) / double(videoProcessor.parameters.minCountForTemplateAveraging));
		double angle = q*2*M_PI;
		int width = int(400.0*q);
		int height = (videoProcessor.lastSegmentationResult.irisCircle.radius-videoProcessor.lastSegmentationResult.pupilCircle.radius)/2 + 1;
		vector< pair<Point, Point> > pts = tools::iterateIris(videoProcessor.lastSegmentationResult, width, height, -M_PI/2, angle-M_PI/2);
		for (size_t i = 0; i < pts.size(); i++) {
			Point p = pts[i].second;
			Vec3f val = image.at<Vec3b>(p);
			Vec3f color;
			double alpha;
			if (status == VideoProcessor::FINISHED_CAPTURE) {
				color =  Vec3f(0,128,0);
				alpha = 0.2;
			} else {
				alpha = 0.8;
				color = Vec3f(0,255,255);
			}
			Vec3b final = Vec3f( val[0]*alpha+color[0]*(1.0-alpha), val[1]*alpha+color[1]*(1.0-alpha), val[2]*alpha+color[2]*(1.0-alpha) );

			image.at<Vec3b>(p) = final;
		}
	}

}
