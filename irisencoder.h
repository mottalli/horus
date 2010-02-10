/*
 * irisencoder.h
 *
 *  Created on: Jun 10, 2009
 *      Author: marcelo
 */

#pragma once

#include "common.h"
#include "segmentationresult.h"
#include "loggabor1dfilter.h"
#include "iristemplate.h"

class IrisEncoder {
public:
	static const double THETA0 = -M_PI/4.0;
	static const double THETA1 = (5.0/4.0) * M_PI;
	static const double RADIUS_TO_USE = 0.75;
	
	IrisEncoder();
	virtual ~IrisEncoder();

	IrisTemplate generateTemplate(const Image* image, const SegmentationResult& segmentationResult);

	Image* normalizedTexture;
	CvMat* normalizedNoiseMask;
	Image* resizedTexture;
	CvMat* resizedNoiseMask;

protected:
	static void normalizeIris(const Image* image, Image* dest, CvMat* destMask, const SegmentationResult& segmentationResult);
	void initializeBuffers(const Image* image);
	void extendMask();

	virtual IrisTemplate encodeTexture(const Image* texture, const CvMat* mask) = 0;
};

