/*
 * decorator.cpp
 *
 *  Created on: Jun 4, 2009
 *      Author: marcelo
 */

#include "decorator.h"

Decorator::Decorator() {

}

void Decorator::drawSegmentationResult(Image* image, const SegmentationResult& segmentationResult)
{
	this->drawContour(image, segmentationResult.irisContour);
	this->drawContour(image, segmentationResult.pupilContour);
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
			cvLine(image, lastPoint, p, CV_RGB(255,255,255), 1);
		}
		lastPoint = p;
	}

	cvLine(image, lastPoint, p0, CV_RGB(255,255,255), 1);
}
