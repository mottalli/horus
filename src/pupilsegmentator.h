/*
 * File:   pupilsegmentator.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:39 PM
*/

#pragma once

#include "common.h"

namespace horus {

class PupilSegmentatorParameters
{
public:
	int bufferWidth;
	double muPupil;
	double sigmaPupil;
	int minimumPupilRadius;
	int maximumPupilRadius;
	int pupilAdjustmentRingWidth;
	int pupilAdjustmentRingHeight;
	int infraredThreshold;
	bool avoidPupilReflection;

	PupilSegmentatorParameters()
	{
		this->bufferWidth = 320;
		this->muPupil = 0.0;
		this->sigmaPupil = 2.0;
		this->minimumPupilRadius = 7;
		this->maximumPupilRadius = 80;
		this->pupilAdjustmentRingWidth = 256;
		this->pupilAdjustmentRingHeight = 100;
		this->infraredThreshold = 200;
		this->avoidPupilReflection = true;
	}
};

class PupilSegmentator {
public:
	PupilSegmentator();
	virtual ~PupilSegmentator();

	ContourAndCloseCircle segmentPupil(const GrayscaleImage& image);
	inline int getPupilContourQuality() const { return this->pupilContourQuality; }

	inline void setROI(Rect ROI) { this->eyeROI = ROI; }
	inline void unsetROI() { this->eyeROI = Rect(); }
	inline bool hasROI() const { return this->eyeROI.width > 0; }

	// Internal buffers
	GrayscaleImage similarityImage;
	GrayscaleImage equalizedImage;
	GrayscaleImage adjustmentRing;
	Mat_<int16_t> adjustmentRingGradient;
	GrayscaleImage workingImage;
	Mat1f adjustmentSnake;
	Mat1f originalAdjustmentSnake;
	GrayscaleImage _LUT;
	double resizeFactor;

	PupilSegmentatorParameters parameters;

	Rect eyeROI;

private:
	void setupBuffers(const Image& image);
	void similarityTransform();
	Circle approximatePupil(const GrayscaleImage& image);
	Circle cascadedIntegroDifferentialOperator(const GrayscaleImage& image, Rect ROI=Rect());
	int calculatePupilContourQuality(const GrayscaleImage& region, const Mat_<uint16_t>& regionGradient, const Mat_<float>& contourSnake);

	int pupilContourQuality;

	typedef struct {
		int maxRad;
		int maxStep;
	} MaxAvgRadiusResult;
	MaxAvgRadiusResult maxAvgRadius(const GrayscaleImage& image, int x, int y, int radmin, int radmax, int radstep);

	uint8_t circleAverage(const GrayscaleImage& image, int x, int y, int radius);
	Contour adjustPupilContour(const GrayscaleImage& image, const Circle& approximateCircle);

	double _lastSigma, _lastMu;
	GrayscaleImage matStructElem;
};


};
