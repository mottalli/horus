#pragma once

#include "common.h"
#include "segmentationresult.h"
#include "iristemplate.h"

/**
 * Abstract class -- must implement texture encoding algorithm
 */
class IrisEncoder {
public:
	static const double THETA0;
	static const double THETA1;
	static const double RADIUS_TO_USE;
	
	IrisEncoder();
	virtual ~IrisEncoder();

	IrisTemplate generateTemplate(const Mat& image, const SegmentationResult& segmentationResult);

protected:
	static void normalizeIris(const Mat_<uint8_t>& image, Mat_<uint8_t>& dest, Mat_<uint8_t>& destMask, const SegmentationResult& segmentationResult, double theta0=THETA0, double theta1=THETA1, double radius=RADIUS_TO_USE);
	static Size getOptimumTemplateSize(int width, int height);		// Returns the optimum template size that is closer to (width, height)

	Mat_<uint8_t> normalizedTexture;
	Mat_<uint8_t> normalizedNoiseMask;

	void extendMask();

	virtual IrisTemplate encodeTexture(const Mat_<uint8_t>& texture, const Mat_<uint8_t>& mask) = 0;
};

