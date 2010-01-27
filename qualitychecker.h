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

	double interlacedCorrelation(const Image* image);
	double checkFocus(const Image* image);
	double segmentationScore(const Image* image, const SegmentationResult& segmentationResult);
	bool checkIrisQuality(const Image* image, const SegmentationResult& segmentationResult);
};

