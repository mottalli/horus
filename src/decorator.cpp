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

void Decorator::drawSegmentationResult(IplImage* image, const SegmentationResult& segmentationResult)
{
	this->drawContour(image, segmentationResult.irisContour, this->irisColor);
	this->drawContour(image, segmentationResult.pupilContour, this->pupilColor);

	Circle irisCircle = segmentationResult.irisCircle;

	if (segmentationResult.eyelidsSegmented) {
		int xMin = segmentationResult.irisCircle.xc-segmentationResult.irisCircle.radius;
		int xMax = segmentationResult.irisCircle.xc+segmentationResult.irisCircle.radius;
		this->drawParabola(image, segmentationResult.upperEyelid, xMin, xMax, this->upperEyelidColor);
		this->drawParabola(image, segmentationResult.lowerEyelid, xMin, xMax, this->lowerEyelidColor);
	}

	/*cvCircle(image, cvPoint(segmentationResult.irisCircle.xc,segmentationResult.irisCircle.yc), segmentationResult.irisCircle.radius, CV_RGB(255,255,255), 1);
	cvCircle(image, cvPoint(segmentationResult.pupilCircle.xc,segmentationResult.pupilCircle.yc), segmentationResult.pupilCircle.radius, CV_RGB(255,255,255), 1);*/
}

void Decorator::drawEncodingZone(IplImage* image, const SegmentationResult& segmentationResult)
{
	bool fill = false;
	Parameters* parameters = Parameters::getParameters();

	int width = parameters->normalizationWidth, height = parameters->normalizationHeight;

	std::vector< std::pair<CvPoint, CvPoint> > irisPoints = Tools::iterateIris(segmentationResult,
		width, height, IrisEncoder::THETA0,
		IrisEncoder::THETA1, IrisEncoder::RADIUS_TO_USE);

	for (size_t i = 0; i < irisPoints.size(); i++) {
		CvPoint imagePoint = irisPoints[i].second;
		CvPoint coord = irisPoints[i].first;
		int x = int(imagePoint.x), y = int(imagePoint.y);

		if (x < 0 || x >= image->width || y < 0 || y > image->height) {
			continue;
		}

		if (fill || (coord.x == 0 || coord.x == width-1 || coord.y == 0 || coord.y == height-1)) {
			cvSet2D(image, y, x, CV_RGB(255,255,0));
		}
	}

}


void Decorator::drawContour(IplImage* image, const Contour& contour, CvScalar color)
{
	if (contour.size() < 2) return;

	const CvPoint p0 = contour[0];
	int n = contour.size();

	CvPoint lastPoint = p0;

	for (int i = 1; i < n; i++) {
		const CvPoint p = contour[i];

		if (false) {

		} else {
			cvLine(image, lastPoint, p, color, 1);
		}
		lastPoint = p;
	}

	cvLine(image, lastPoint, p0, color, 1);
}

void Decorator::drawParabola(IplImage* image, const Parabola& parabola, int xMin, int xMax, CvScalar color)
{
	if (xMin < 0) xMin = 1;
	if (xMax < 0 || xMax >= image->width) xMax = image->width-1;

	CvPoint lastPoint = cvPoint(xMin, int(parabola.value(xMin)));
	for (int x = xMin+1; x <= xMax; x++) {
		CvPoint point = cvPoint(x, int(parabola.value(x)));

		cvLine(image, lastPoint, point, color, 1);
		lastPoint = point;
	}
}

void Decorator::drawTemplate(IplImage* image, const IrisTemplate& irisTemplate)
{
	IplImage* imgTemplate = irisTemplate.getTemplateImage();
	IplImage* imgMask = irisTemplate.getNoiseMaskImage();

	// Joins the template and the mask in the same image as follows:
	// mask = 0 => res = 0
	// template = 1 => res = 255
	// template = 0 => res = 128
	cvThreshold(imgTemplate, imgTemplate, 128, 0 /* notused */, CV_THRESH_TRUNC);
	cvAddS(imgTemplate, cvScalar(128), imgTemplate, imgMask);

	CvSize size = cvGetSize(imgTemplate);

	CvMat region;
	CvPoint topleftTemplate = cvPoint(10, 10);

	cvGetSubRect(image, &region, cvRect(topleftTemplate.x, topleftTemplate.y, size.width, size.height));
	if (image->nChannels == 3) {
		cvMerge(imgTemplate, NULL, NULL, NULL, &region);
		cvMerge(NULL, imgTemplate, NULL, NULL, &region);
		cvMerge(NULL, NULL, imgTemplate, NULL, &region);
	} else {
		cvCopy(imgTemplate, &region);
	}
	cvRectangle(image, topleftTemplate, cvPoint(topleftTemplate.x+size.width-1, topleftTemplate.y+size.height-1), CV_RGB(0,0,0), 1);

	cvReleaseImage(&imgTemplate);
	cvReleaseImage(&imgMask);
}
