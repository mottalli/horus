/*
 * File:   pupilsegmentator.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:39 PM
*/

#pragma once

#include "common.h"

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
	}
};

class PupilSegmentator {
public:
	PupilSegmentator();
	virtual ~PupilSegmentator();

	ContourAndCloseCircle segmentPupil(const GrayscaleImage& image);
	inline int getPupilContourQuality() const { return this->pupilContourQuality; }

	inline void setROI(Rect ROI) { this->ROI = ROI; }
	inline void unsetROI() { this->ROI = Rect(0,0,0,0); }

	// Internal buffers
	GrayscaleImage similarityImage;
	GrayscaleImage equalizedImage;
	GrayscaleImage adjustmentRing;
	Mat_<int16_t> adjustmentRingGradient;
	GrayscaleImage workingImage;
	Mat_<float> adjustmentSnake;
	Mat_<float> originalAdjustmentSnake;
	GrayscaleImage _LUT;
	double resizeFactor;

	PupilSegmentatorParameters parameters;

private:
	void setupBuffers(const Image& image);
	void similarityTransform();
	Circle approximatePupil(const GrayscaleImage& image);
	Circle cascadedIntegroDifferentialOperator(const GrayscaleImage& image);
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

	Rect ROI, workingROI;
};


