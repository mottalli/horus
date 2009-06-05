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
}

void Decorator::drawSegmentationResult(Image* image, const SegmentationResult& segmentationResult)
{
	this->drawContour(image, segmentationResult.irisContour);
	this->drawContour(image, segmentationResult.pupilContour);

	Circle irisCircle = segmentationResult.irisCircle;

	int xMin = std::max(0, irisCircle.xc - int(1.5*irisCircle.radius));
	int xMax = std::min(image->width, irisCircle.xc + int(1.5*irisCircle.radius));

	this->drawParabola(image, segmentationResult.upperEyelid, -1, -1);
	this->drawParabola(image, segmentationResult.lowerEyelid, -1, -1);

	/*cvCircle(image, cvPoint(segmentationResult.irisCircle.xc,segmentationResult.irisCircle.yc), segmentationResult.irisCircle.radius, CV_RGB(255,255,255), 1);
	cvCircle(image, cvPoint(segmentationResult.pupilCircle.xc,segmentationResult.pupilCircle.yc), segmentationResult.pupilCircle.radius, CV_RGB(255,255,255), 1);*/
}

void Decorator::drawContour(Image* image, const Contour& contour)
{
	const CvPoint p0 = contour[0];
	int n = contour.size();

	CvPoint lastPoint = p0;

	for (int i = 1; i < n; i++) {
		const CvPoint p = contour[i];

		if (false) {

		} else {
			cvLine(image, lastPoint, p, CV_RGB(255,0,0), 1);
		}
		lastPoint = p;
	}

	cvLine(image, lastPoint, p0, CV_RGB(255,0,0), 1);
}

void Decorator::drawParabola(Image* image, const Parabola& parabola, int xMin, int xMax)
{
	if (xMin == -1) xMin = 1;
	if (xMax == -1) xMax = image->width;

	CvPoint lastPoint = cvPoint(xMin, int(parabola.value(xMin)));
	for (int x = xMin+1; x <= xMax; x++) {
		CvPoint point = cvPoint(x, int(parabola.value(x)));
		cvLine(image, lastPoint, point, CV_RGB(255,0,0), 1);
		lastPoint = point;
	}
}
