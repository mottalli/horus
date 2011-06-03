#pragma once

#include "common.h"
#include "iristemplate.h"

/**
 * Abstract class -- must implement texture encoding algorithm
 */

namespace horus {

class IrisEncoder {
public:
	static const double THETA0;
	static const double THETA1;
	static const double MIN_RADIUS_TO_USE;
	static const double MAX_RADIUS_TO_USE;
	
	IrisEncoder();
	virtual ~IrisEncoder();

	IrisTemplate generateTemplate(const Image& image, const SegmentationResult& segmentationResult);
	static void normalizeIris(const GrayscaleImage& image, GrayscaleImage& dest, GrayscaleImage& destMask, const SegmentationResult& segmentationResult, double theta0 = 0.0, double theta1 = 2.0*M_PI, double radiusMin = 0.0, double radiusMax = 1.0);

	virtual string getEncoderSignature() const = 0;

	/**
	 * Given a list of templates, returns an "averaged" template
	 */
	static IrisTemplate averageTemplates(const vector<const IrisTemplate*>& templates);

	GrayscaleImage normalizedTexture;

protected:
	static Size getOptimumTemplateSize(int width, int height);		// Returns the optimum template size that is closer to (width, height)

	GrayscaleImage normalizedNoiseMask;

	void extendMask();

	virtual IrisTemplate encodeTexture(const GrayscaleImage& texture, const GrayscaleImage& mask) = 0;
	virtual Size getNormalizationSize();
};

}
