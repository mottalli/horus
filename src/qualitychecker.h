#pragma once

#include "common.h"
#include "segmentationresult.h"
#include "iristemplate.h"
#include "templatecomparator.h"

class QualityCheckerParameters
{
public:
	int pupilIrisZScore;
	int pupilIrisGrayDiff;

	QualityCheckerParameters()
	{
		this->pupilIrisGrayDiff = 20;
		this->pupilIrisZScore = 3;
	}
};

class QualityChecker {
public:
	QualityChecker();
	virtual ~QualityChecker();

	double interlacedCorrelation(const GrayscaleImage& image);
	double checkFocus(const Image& image);
	double getIrisQuality(const GrayscaleImage& image, const SegmentationResult& segmentationResult);

	typedef enum {
		NO_COUNT,
		LOW_CONTRAST,
		LOW_ZSCORE,
		HAS_IRIS
	} ValidationHeuristics;
	ValidationHeuristics validateIris(const GrayscaleImage& image, const SegmentationResult& segmentationResult);

	QualityCheckerParameters parameters;

	//TODO
	double irisTemplateQuality(const IrisTemplate& irisTemplate);
	double matchQuality(const TemplateComparator& comparator);

private:
	GrayscaleImage evenFrame, oddFrame;
	Mat_<float> bufX, bufY, bufMul;
	Mat bufSobel;
};

