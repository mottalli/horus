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

	static const Scalar DEFAULT_IRIS_COLOR, DEFAULT_PUPIL_COLOR, DEFAULT_EYELID_COLOR;

	void drawSegmentationResult(Mat& image, const SegmentationResult& segmentationResult) const;
	void drawTemplate(Mat& image, const IrisTemplate& irisTemplate);
	void drawEncodingZone(Mat& image, const SegmentationResult& segmentationResult);

	void setDrawingColors(Scalar pupilColor = DEFAULT_PUPIL_COLOR,
							Scalar irisColor = DEFAULT_IRIS_COLOR,
							Scalar upperEyelidColor = DEFAULT_EYELID_COLOR,
							Scalar lowerEyelidColor = DEFAULT_EYELID_COLOR
						  );
	void drawFocusScores(const list<double>& focusScores, Mat image, Rect rect, double threshold);

	int lineWidth;

private:
	void drawContour(Mat& image, const Contour& contour, const Scalar& color) const;
	void drawParabola(Mat& image, const Parabola& parabola, int xMin, int xMax, const Scalar& color) const;
};

