/*
 * irisencoder.h
 *
 *  Created on: Jun 10, 2009
 *      Author: marcelo
 */

#ifndef IRISENCODER_H_
#define IRISENCODER_H_

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

	struct {
		Image* normalizedTexture;
		Image* resizedTexture;
		CvMat* normalizedNoiseMask;
		CvMat* resizedNoiseMask;

		Image* filteredTexture;
		Image* filteredTextureReal;
		Image* filteredTextureImag;
		CvMat* thresholdedTexture;
	} buffers;

	// For debugging purposes
	const Image* getNormalizedTexture() const { return this->buffers.normalizedTexture; }

protected:
	static void normalizeIris(const Image* image, Image* dest, CvMat* destMask, const SegmentationResult& segmentationResult);
	void initializeBuffers(const Image* image);
	LogGabor1DFilter filter;
	void applyFilter();
	void extendMask();
};

#endif /* IRISENCODER_H_ */
