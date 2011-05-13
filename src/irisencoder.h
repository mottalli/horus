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

	IrisTemplate generateTemplate(const Image& image, const SegmentationResult& segmentationResult);
	static void normalizeIris(const GrayscaleImage& image, GrayscaleImage& dest, GrayscaleImage& destMask, const SegmentationResult& segmentationResult, double theta0 = 0.0, double theta1 = 2.0*M_PI, double radius = 1.0);

	virtual string getEncoderSignature() const = 0;

	/**
	 * Given a list of templates, returns an "averaged" template
	 */
	static IrisTemplate averageTemplates(const vector<const IrisTemplate*>& templates);

protected:
	static Size getOptimumTemplateSize(int width, int height);		// Returns the optimum template size that is closer to (width, height)

	GrayscaleImage normalizedTexture;
	GrayscaleImage normalizedNoiseMask;

	void extendMask();

	virtual IrisTemplate encodeTexture(const GrayscaleImage& texture, const GrayscaleImage& mask) = 0;
	virtual Size getNormalizationSize();
};

