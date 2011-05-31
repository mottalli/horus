#pragma once

#include "common.h"
#include "segmentationresult.h"
#include "iristemplate.h"
#include "videoprocessor.h"

class Decorator {
public:
	Decorator();

	Scalar pupilColor;
	Scalar irisColor;
	Scalar upperEyelidColor;
	Scalar lowerEyelidColor;

	static const Scalar DEFAULT_IRIS_COLOR, DEFAULT_PUPIL_COLOR, DEFAULT_EYELID_COLOR;

	void drawSegmentationResult(Image& image, const SegmentationResult& segmentationResult) const;
	void drawTemplate(Image& image, const IrisTemplate& irisTemplate, Point p0=Point(15,15));
	void drawEncodingZone(Image& image, const SegmentationResult& segmentationResult);

	void setDrawingColors(Scalar pupilColor = DEFAULT_PUPIL_COLOR,
							Scalar irisColor = DEFAULT_IRIS_COLOR,
							Scalar upperEyelidColor = DEFAULT_EYELID_COLOR,
							Scalar lowerEyelidColor = DEFAULT_EYELID_COLOR
						  );
	void drawFocusScores(Image& image, const list<double>& focusScores, Rect rect, double threshold);
	void drawIrisTexture(const Image& imageSrc, Image& imageDest, SegmentationResult segmentationResult);
	void drawCaptureStatus(Image& image, const VideoProcessor& videoProcessor);

	int lineWidth;

private:
	void drawContour(Image& image, const Contour& contour, const Scalar& color) const;
	void drawParabola(Image& image, const Parabola& parabola, int xMin, int xMax, const Scalar& color) const;
};

