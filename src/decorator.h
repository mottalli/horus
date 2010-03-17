/*
 * decorator.h
 *
 *  Created on: Jun 4, 2009
 *      Author: marcelo
 */

#pragma once

#include "common.h"
#include "segmentationresult.h"
#include "iristemplate.h"

class Decorator {
public:
	Decorator();

	CvScalar pupilColor;
	CvScalar irisColor;
	CvScalar upperEyelidColor;
	CvScalar lowerEyelidColor;

	void drawSegmentationResult(IplImage* image, const SegmentationResult& segmentationResult);
	void drawTemplate(IplImage* image, const IrisTemplate& irisTemplate);
	void drawEncodingZone(IplImage* image, const SegmentationResult& segmentationResult);
private:
	void drawContour(IplImage* image, const Contour& contour, CvScalar color);
	void drawParabola(IplImage* image, const Parabola& parabola, int xMin, int xMax, CvScalar color);
};

