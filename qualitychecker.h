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
	bool validateIris(const Image* image, const SegmentationResult& segmentationResult);
	double getIrisQuality(const Image* image, const SegmentationResult& segmentationResult);
};

