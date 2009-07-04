/*
 * decorator.cpp
 *
 *  Created on: Jun 4, 2009
 *      Author: marcelo
 */

#include "decorator.h"
#include <cmath>

Decorator::Decorator()
{
	this->irisColor = CV_RGB(0,255,0);
	this->pupilColor = CV_RGB(255,0,0);
	this->upperEyelidColor = CV_RGB(0,0,255);
	this->lowerEyelidColor = CV_RGB(0,0,255);
}

void Decorator::drawSegmentationResult(Image* image, const SegmentationResult& segmentationResult)
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

void Decorator::drawContour(Image* image, const Contour& contour, CvScalar color)
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

void Decorator::drawParabola(Image* image, const Parabola& parabola, int xMin, int xMax, CvScalar color)
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
