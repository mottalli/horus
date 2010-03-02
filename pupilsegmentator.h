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

	ContourAndCloseCircle segmentPupil(const IplImage* image);
	inline int getPupilContourQuality() const { return this->pupilContourQuality; }

	// Internal buffers
	IplImage* similarityImage;
	IplImage* equalizedImage;
	IplImage* adjustmentRing;
	IplImage* adjustmentRingGradient;
	IplImage* workingImage;
	CvMat* adjustmentSnake;
	CvMat* LUT;
	double resizeFactor;

private:
	void setupBuffers(const IplImage* image);
	void similarityTransform();
	Circle approximatePupil(const IplImage* image);
	Circle cascadedIntegroDifferentialOperator(const IplImage* image);
	int calculatePupilContourQuality(const IplImage* region, const IplImage* regionGradient, const CvMat* contourSnake);

	int pupilContourQuality;

	typedef struct {
		int maxRad;
		int maxStep;
	} MaxAvgRadiusResult;
	MaxAvgRadiusResult maxAvgRadius(const IplImage* image, int x, int y, int radmin, int radmax, int radstep);

	uint8_t circleAverage(const IplImage* image, int x, int y, int radius);
	Contour adjustPupilContour(const IplImage* image, const Circle& approximateCircle);

	double _lastSigma, _lastMu;
};


