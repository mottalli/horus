/*
 * File:   pupilsegmentator.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:39 PM
*/

#pragma once

#include "common.h"


class PupilSegmentator {
public:
	PupilSegmentator();
	virtual ~PupilSegmentator();

	ContourAndCloseCircle segmentPupil(const Image* image);
	inline int getPupilContourQuality() const { return this->pupilContourQuality; }

	// Internal buffers
	IplImage* similarityImage;
	Image* equalizedImage;
	Image* adjustmentRing;
	Image* adjustmentRingGradient;
	Image* workingImage;
	CvMat* adjustmentSnake;
	CvMat* LUT;
	double resizeFactor;

private:
	void setupBuffers(const Image* image);
	void similarityTransform();
	Circle approximatePupil(const Image* image);
	Circle cascadedIntegroDifferentialOperator(const Image* image);
	int calculatePupilContourQuality(const Image* region, const Image* regionGradient, const CvMat* contourSnake);

	int pupilContourQuality;

	typedef struct {
		int maxRad;
		int maxStep;
	} MaxAvgRadiusResult;
	MaxAvgRadiusResult maxAvgRadius(const Image* image, int x, int y, int radmin, int radmax, int radstep);

	uint8_t circleAverage(const Image* image, int x, int y, int radius);
	Contour adjustPupilContour(const Image* image, const Circle& approximateCircle);

	double _lastSigma, _lastMu;
};


