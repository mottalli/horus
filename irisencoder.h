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

class IrisEncoder {
public:
	IrisEncoder();
	virtual ~IrisEncoder();

	void generateTemplate(const Image* image, const SegmentationResult& segmentationResult);
	static void normalizeIris(const Image* image, Image* dest, CvMat* destMask, const SegmentationResult& segmentationResult);

	struct {
		Image* normalizedTexture;
		CvMat* noiseMask;
		Image* filteredTexture;
	} buffers;

private:
	void initializeBuffers(const Image* image);
	LogGabor1DFilter filter;
	void applyFilter();
};

#endif /* IRISENCODER_H_ */
