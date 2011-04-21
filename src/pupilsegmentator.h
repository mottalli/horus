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

	ContourAndCloseCircle segmentPupil(const Mat& image);
	inline int getPupilContourQuality() const { return this->pupilContourQuality; }

	inline void setROI(Rect ROI) { this->ROI = ROI; };
	inline void unsetROI() { this->ROI = Rect(0,0,0,0); };

	// Internal buffers
	Mat_<uint8_t> similarityImage;
	Mat_<uint8_t> equalizedImage;
	Mat_<uint8_t> adjustmentRing;
	Mat_<int16_t> adjustmentRingGradient;
	Mat_<uint8_t> workingImage;
	Mat_<float> adjustmentSnake;
	Mat_<float> originalAdjustmentSnake;
	Mat_<uint8_t> _LUT;
	double resizeFactor;

private:
	void setupBuffers(const Mat& image);
	void similarityTransform();
	Circle approximatePupil(const Mat_<uint8_t>& image);
	Circle cascadedIntegroDifferentialOperator(const Mat_<uint8_t>& image);
	int calculatePupilContourQuality(const Mat_<uint8_t>& region, const Mat_<uint16_t>& regionGradient, const Mat_<float>& contourSnake);

	int pupilContourQuality;

	typedef struct {
		int maxRad;
		int maxStep;
	} MaxAvgRadiusResult;
	MaxAvgRadiusResult maxAvgRadius(const Mat_<uint8_t>& image, int x, int y, int radmin, int radmax, int radstep);

	uint8_t circleAverage(const Mat_<uint8_t>& image, int x, int y, int radius);
	Contour adjustPupilContour(const Mat_<uint8_t>& image, const Circle& approximateCircle);

	double _lastSigma, _lastMu;

	Rect ROI;
};


