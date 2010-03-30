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

	void drawSegmentationResult(Mat& image, const SegmentationResult& segmentationResult) const;
	void drawTemplate(IplImage* image, const IrisTemplate& irisTemplate);
	void drawEncodingZone(Mat& image, const SegmentationResult& segmentationResult);
private:
	void drawContour(Mat& image, const Contour& contour, const Scalar& color) const;
	void drawParabola(IplImage* image, const Parabola& parabola, int xMin, int xMax, CvScalar color);
};

