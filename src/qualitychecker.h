#pragma once

#include "common.h"
#include "segmentationresult.h"

class QualityChecker {
public:
	QualityChecker();
	virtual ~QualityChecker();

	double interlacedCorrelation(const Mat& image);
	double checkFocus(const Mat& image);
	double getIrisQuality(const Mat& image, const SegmentationResult& segmentationResult);

	typedef enum {
		NO_COUNT,
		LOW_CONTRAST,
		LOW_ZSCORE,
		HAS_IRIS
	} ValidationHeuristics;
	ValidationHeuristics validateIris(const Mat& image, const SegmentationResult& segmentationResult);

	struct {
		int pupilIrisZScore;
		int pupilIrisGrayDiff;
	} parameters;

//private:
	Mat evenFrame, oddFrame;
	Mat_<float> bufX, bufY, bufMul;
	Mat bufSobel;
};

