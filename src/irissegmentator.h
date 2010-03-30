#pragma once

#include "common.h"

class IrisSegmentator {
public:
    IrisSegmentator();
    virtual ~IrisSegmentator();

	Mat_<uint8_t> adjustmentRing;
	Mat_<int16_t> adjustmentRingGradient;
	Mat_<float> adjustmentSnake;

	ContourAndCloseCircle segmentIris(const Mat& image, const ContourAndCloseCircle& pupilSegmentation);

private:
	void setupBuffers(const Mat_<uint8_t>& image);
	ContourAndCloseCircle segmentIrisRecursive(const Mat_<uint8_t>& image, const ContourAndCloseCircle& pupilSegmentation, int radiusMax=-1, int radiusMin=-1);
};


