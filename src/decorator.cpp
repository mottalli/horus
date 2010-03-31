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
	this->irisColor = CV_RGB(0,255,0);
	this->pupilColor = CV_RGB(255,0,0);
	this->upperEyelidColor = CV_RGB(0,0,255);
	this->lowerEyelidColor = CV_RGB(0,0,255);
}

void Decorator::drawSegmentationResult(Mat& image, const SegmentationResult& segmentationResult) const
{
	Scalar irisColor_ = (image.channels() == 1 ? CV_RGB(255,255,255) : this->irisColor);
	Scalar pupilColor_ = (image.channels() == 1 ? CV_RGB(255,255,255) : this->pupilColor);
	this->drawContour(image, segmentationResult.irisContour, irisColor_);
	this->drawContour(image, segmentationResult.pupilContour, pupilColor_);

	const Circle& irisCircle = segmentationResult.irisCircle;

	if (segmentationResult.eyelidsSegmented) {
		int xMin = segmentationResult.irisCircle.xc-segmentationResult.irisCircle.radius;
		int xMax = segmentationResult.irisCircle.xc+segmentationResult.irisCircle.radius;
		//this->drawParabola(image, segmentationResult.upperEyelid, xMin, xMax, this->upperEyelidColor);
		//this->drawParabola(image, segmentationResult.lowerEyelid, xMin, xMax, this->lowerEyelidColor);
	}
}

void Decorator::drawEncodingZone(Mat& image, const SegmentationResult& segmentationResult)
{
	bool fill = false;
	Parameters* parameters = Parameters::getParameters();

	int width = parameters->normalizationWidth, height = parameters->normalizationHeight;

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
			if (image.channels() == 1) {
				image.at<uchar>(y,x) = 255;
			} else if (image.channels() == 3) {
				image.at<Vec3b>(y, x) = Vec3b(0,255,255);
			}
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

void Decorator::drawParabola(IplImage* image, const Parabola& parabola, int xMin, int xMax, CvScalar color)
{
	if (xMin < 0) xMin = 1;
	if (xMax < 0 || xMax >= image->width) xMax = image->width-1;

	Point lastPoint = Point(xMin, int(parabola.value(xMin)));
	for (int x = xMin+1; x <= xMax; x++) {
		Point point = Point(x, int(parabola.value(x)));

		cvLine(image, lastPoint, point, color, 1);
		lastPoint = point;
	}
}

void Decorator::drawTemplate(Mat& image, const IrisTemplate& irisTemplate)
{
	IplImage* imgTemplate = irisTemplate.getTemplateImage();
	CvMat* mask = irisTemplate.getUnpackedMask();

	cvSet(mask, cvScalar(255), mask);
	cvNot(mask, mask);					// Hacky way to NOT the template
	cvSet(imgTemplate, cvScalar(127), mask);

	IplImage* decoratedTemplate;

	if (imgTemplate->height < 10) {
		IplImage* tmp = cvCreateImage(cvSize(imgTemplate->width+2, 3*imgTemplate->height+2), IPL_DEPTH_8U, 1);
		cvSet(tmp, cvScalar(128));
		int width = imgTemplate->width;
		for (int i = 0; i < imgTemplate->height; i++) {
			CvMat tmpsrc, tmpdest;
			cvGetSubRect(imgTemplate, &tmpsrc, cvRect(0, i, width, 1));
			cvGetSubRect(tmp, &tmpdest, cvRect(1, 3*i+2, width, 1));
			cvCopy(&tmpsrc, &tmpdest);
		}
		decoratedTemplate = cvCreateImage(cvSize(1.5*tmp->width, 2.5*tmp->height), IPL_DEPTH_8U, 1);
		cvResize(tmp, decoratedTemplate, CV_INTER_NN);
		cvReleaseImage(&tmp);
	} else {
		decoratedTemplate = cvCloneImage(imgTemplate);
	}

	CvMat region;
	Point topleftTemplate = Point(10, 10);
	CvSize size = cvGetSize(decoratedTemplate);

	cvGetSubRect(TO_IPLIMAGE(image), &region, cvRect(topleftTemplate.x, topleftTemplate.y, size.width, size.height));
	if (image.channels() == 3) {
		cvMerge(decoratedTemplate, NULL, NULL, NULL, &region);
		cvMerge(NULL, decoratedTemplate, NULL, NULL, &region);
		cvMerge(NULL, NULL, decoratedTemplate, NULL, &region);
	} else {
		cvCopy(decoratedTemplate, &region);
	}
	cvRectangle(TO_IPLIMAGE(image), topleftTemplate, Point(topleftTemplate.x+size.width-1, topleftTemplate.y+size.height-1), CV_RGB(0,0,0), 1);

	cvReleaseImage(&imgTemplate);
	cvReleaseMat(&mask);
	cvReleaseImage(&decoratedTemplate);
}
