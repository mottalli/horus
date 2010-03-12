/*
 * qualitychecker.h
 *
 *  Created on: Jun 19, 2009
 *      Author: marcelo
 */

#pragma once

#include "common.h"
#include "segmentationresult.h"

class QualityChecker {
public:
	QualityChecker();
	virtual ~QualityChecker();

	double interlacedCorrelation(const IplImage* image);
	double checkFocus(const IplImage* image);
	bool validateIris(const IplImage* image, const SegmentationResult& segmentationResult);
	double getIrisQuality(const IplImage* image, const SegmentationResult& segmentationResult);
};

