#pragma once

#include "common.h"

class IrisSegmentatorParameters {
public:
	int irisAdjustmentRingWidth;
	int irisAdjustmentRingHeight;

	IrisSegmentatorParameters()
	{
		this->irisAdjustmentRingWidth = 512;
		this->irisAdjustmentRingHeight = 90;
	}
} ;


class IrisSegmentator {
public:
    IrisSegmentator();
    virtual ~IrisSegmentator();

	GrayscaleImage adjustmentRing;
	Mat_<int16_t> adjustmentRingGradient;
	Mat_<float> adjustmentSnake;

	ContourAndCloseCircle segmentIris(const GrayscaleImage& image, const ContourAndCloseCircle& pupilSegmentation);

	IrisSegmentatorParameters parameters;

private:
	void setupBuffers(const GrayscaleImage& image);
	ContourAndCloseCircle segmentIrisRecursive(const GrayscaleImage& image, const ContourAndCloseCircle& pupilSegmentation, int radiusMax=-1, int radiusMin=-1);
};


