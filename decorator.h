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

	void drawSegmentationResult(Image* image, const SegmentationResult& segmentationResult);
	void drawTemplate(Image* image, const IrisTemplate& irisTemplate);
	void drawEncodingZone(Image* image, const SegmentationResult& segmentationResult);
private:
	void drawContour(Image* image, const Contour& contour, CvScalar color);
	void drawParabola(Image* image, const Parabola& parabola, int xMin, int xMax, CvScalar color);
};

