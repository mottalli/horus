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
	static void normalizeIris(const Mat& image, Mat& dest, Mat& destMask, const SegmentationResult& segmentationResult, double theta0 = 0.0, double theta1 = 2.0*M_PI, double radius = 1.0);

	/**
	 * Given a list of templates,
	 */
	static IrisTemplate averageTemplates(const vector<const IrisTemplate*>& templates);

protected:
	static Size getOptimumTemplateSize(int width, int height);		// Returns the optimum template size that is closer to (width, height)

	Mat_<uint8_t> normalizedTexture;
	Mat_<uint8_t> normalizedNoiseMask;

	void extendMask();

	virtual IrisTemplate encodeTexture(const Mat_<uint8_t>& texture, const Mat_<uint8_t>& mask) = 0;
	virtual Size getNormalizationSize();
};

