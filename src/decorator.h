#pragma once

#include "common.h"
#include "segmentationresult.h"
#include "iristemplate.h"

class Decorator {
public:
	Decorator();

	Scalar pupilColor;
	Scalar irisColor;
	Scalar upperEyelidColor;
	Scalar lowerEyelidColor;

	void drawSegmentationResult(Mat& image, const SegmentationResult& segmentationResult) const;
	void drawTemplate(Mat& image, const IrisTemplate& irisTemplate);
	void drawEncodingZone(Mat& image, const SegmentationResult& segmentationResult);

	void setDrawingColors(Scalar pupilColor = Scalar(0,255,0),
							Scalar irisColor = Scalar(255,0,0),
							Scalar upperEyelidColor = Scalar(0,0,255),
							Scalar lowerEyelidColor = Scalar(0,0,255)
						  );
	void drawFocusScores(const list<double>& focusScores, Mat image, Rect rect, double threshold);

private:
	void drawContour(Mat& image, const Contour& contour, const Scalar& color) const;
	void drawParabola(Mat& image, const Parabola& parabola, int xMin, int xMax, const Scalar& color) const;
};

