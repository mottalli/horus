/*
 * qualitychecker.h
 *
 *  Created on: Jun 19, 2009
 *      Author: marcelo
 */

#ifndef QUALITYCHECKER_H_
#define QUALITYCHECKER_H_

#include "common.h"
#include "segmentationresult.h"

class QualityChecker {
public:
	QualityChecker();
	virtual ~QualityChecker();

	double interlacedCorrelation(const Image* image);
	double checkFocus(const Image* image);
	bool validateSegmentation(const Image* image, const SegmentationResult& segmentationResult);
	bool checkIrisQuality(const Image* image, const SegmentationResult& segmentationResult);
};

#endif /* QUALITYCHECKER_H_ */
