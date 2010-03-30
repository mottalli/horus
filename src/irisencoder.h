/*
 * irisencoder.h
 *
 *  Created on: Jun 10, 2009
 *      Author: marcelo
 */

#pragma once

#include "common.h"
#include "segmentationresult.h"
#include "iristemplate.h"

class IrisEncoder {
public:
	static const double THETA0;
	static const double THETA1;
	static const double RADIUS_TO_USE;
	
	IrisEncoder();
	virtual ~IrisEncoder();

	IrisTemplate generateTemplate(const IplImage* image, const SegmentationResult& segmentationResult);

	IplImage* normalizedTexture;
	CvMat* normalizedNoiseMask;

	static void normalizeIris(const IplImage* image, IplImage* dest, CvMat* destMask, const SegmentationResult& segmentationResult, double theta0=THETA0, double theta1=THETA1, double radius=RADIUS_TO_USE);
	static CvSize getOptimumTemplateSize(int width, int height);		// Returns the optimum template size that is closer to (width, height)
protected:
	void extendMask();

	virtual IrisTemplate encodeTexture(const IplImage* texture, const CvMat* mask) = 0;
};

