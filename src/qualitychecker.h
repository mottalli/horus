#pragma once

#include "common.h"
#include "segmentationresult.h"

class QualityChecker {
public:
	QualityChecker();
	virtual ~QualityChecker();

	double interlacedCorrelation(const Mat& image);
	double checkFocus(const Mat& image);
	bool validateIris(const Mat& image, const SegmentationResult& segmentationResult);
	double getIrisQuality(const Mat& image, const SegmentationResult& segmentationResult);

//private:
	Mat evenFrame, oddFrame;
	Mat_<float> bufX, bufY, bufMul;
	Mat bufSobel;
};

