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
	/*#define THETA0 -M_PI/4.0
	#define THETA1 (5.0/4.0) * M_PI
	#define RADIUS_TO_USE 0.75*/
	
	IrisEncoder();
	virtual ~IrisEncoder();

	IrisTemplate generateTemplate(const IplImage* image, const SegmentationResult& segmentationResult);

	IplImage* normalizedTexture;
	CvMat* normalizedNoiseMask;
	IplImage* resizedTexture;
	CvMat* resizedNoiseMask;

	static void normalizeIris(const IplImage* image, IplImage* dest, CvMat* destMask, const SegmentationResult& segmentationResult, double theta0=THETA0, double theta1=THETA1, double radius=RADIUS_TO_USE);
protected:
	void initializeBuffers(const IplImage* image);
	void extendMask();

	virtual IrisTemplate encodeTexture(const IplImage* texture, const CvMat* mask) = 0;
};

