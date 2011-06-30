#include <cmath>
#include <iostream>

#include "irisencoder.h"
#include "tools.h"

using namespace horus;
using namespace std;

const double IrisEncoder::THETA0 = -M_PI/4.0;
const double IrisEncoder::THETA1 = (5.0/4.0) * M_PI;
const double IrisEncoder::MIN_RADIUS_TO_USE = 0.1;
const double IrisEncoder::MAX_RADIUS_TO_USE = 0.8;

IrisEncoder::IrisEncoder()
{
}

IrisEncoder::~IrisEncoder()
{
}

IrisTemplate IrisEncoder::generateTemplate(const Image& image, const SegmentationResult& segmentationResult)
{
	assert(image.type() == CV_8UC1 || image.type() == CV_8UC3);

	// We can only process grayscale images. If it's a color image, we need to convert it. Try to optimise whenever
	// possible.
	GrayscaleImage bwimage;
	tools::toGrayscale(image, bwimage, false);

	Size normalizedSize = this->getNormalizationSize();
	GrayscaleImage normalizedTexture(normalizedSize), normalizedNoiseMask(normalizedSize);
	IrisEncoder::normalizeIris(bwimage, normalizedTexture, normalizedNoiseMask, segmentationResult, IrisEncoder::THETA0, IrisEncoder::THETA1, IrisEncoder::MIN_RADIUS_TO_USE, IrisEncoder::MAX_RADIUS_TO_USE);

	// Improve the iris mask
	this->extendMask(normalizedTexture, normalizedNoiseMask);

	return this->encodeTexture(normalizedTexture, normalizedNoiseMask);
}

void IrisEncoder::extendMask(const GrayscaleImage& texture, GrayscaleImage& mask)
{
	// Mask away pixels too far from the mean
	Scalar smean, sdev;
	meanStdDev(texture, smean, sdev, mask);

	double mean = smean.val[0], dev = sdev.val[0];
	uint8_t uthresh = uint8_t(mean+1.5*dev);
	uint8_t lthresh = uint8_t(mean-1.5*dev);

	for (int y = 0; y < texture.rows; y++) {
		const uint8_t* row = texture.ptr(y);
		for (int x = 0; x < texture.cols; x++) {
			uint8_t val = row[x];
			if (val < lthresh || val > uthresh) {
				mask(y, x) = 0;
			}
		}
	}
}

void IrisEncoder::normalizeIris(const GrayscaleImage& image, GrayscaleImage& dest, GrayscaleImage& destMask, const SegmentationResult& segmentationResult, double theta0, double theta1, double radiusMin, double radiusMax)
{
	// Initialize the mask to 1 (all bits enabled)
	destMask.setTo(Scalar(1));

	SegmentationResult::iterator it = segmentationResult.iterateIris(dest.size(), theta0, theta1, radiusMin, radiusMax);
	do {
		Point imagePoint = it.imagePoint;
		Point coord = it.texturePoint;

		int ximage0 = imagePoint.x;
		int ximage1 = imagePoint.x+1;
		int yimage0 = imagePoint.y;
		int yimage1 = imagePoint.y+1;

		if (ximage0 < 0 || ximage1 >= image.cols || yimage0 < 0 || yimage1 >= image.rows) {
			dest(coord) = 0;
			destMask(coord) = 0;
		} else {
			unsigned v1 = image(yimage0, ximage0);
			unsigned v2 = image(yimage0, ximage1);
			unsigned v3 = image(yimage1, ximage0);
			unsigned v4 = image(yimage1, ximage1);
			dest(coord) = uint8_t((v1+v2+v3+v4)/4);
		}

		// See if (x,y) is occluded by an eyelid
		if (it.isOccluded) {
			destMask(coord) = 0;
		}
	} while (it.next());
}

Size IrisEncoder::getOptimumTemplateSize(int width, int height)
{
	int optimumWidth = int(ceil(float(width)/32.0)) * 32; // Must be a multiple of 32
	return Size(optimumWidth, height);
}

Size IrisEncoder::getNormalizationSize() const
{
	return Size(512, 80);
}

IrisTemplate IrisEncoder::averageTemplates(const vector<IrisTemplate>& templates)
{
	assert(templates.size() >= 1);
	Mat1b unpackedTemplate, unpackedMask;
	Mat1b acum, acumMask;

	// Just retrieve the size of the templates
	unpackedTemplate = templates[0].getUnpackedTemplate();
	acum = Mat1b::zeros(unpackedTemplate.size());
	acumMask = Mat1b::zeros(unpackedTemplate.size());

	for (size_t i = 0; i < templates.size(); i++) {
		unpackedTemplate = templates[i].getUnpackedTemplate();
		unpackedMask = templates[i].getUnpackedMask();

		assert(acum.size() == unpackedTemplate.size());

		acum += unpackedTemplate;
		acumMask += unpackedMask;
	}

	Mat averageTemplate = Mat::zeros(acum.size(), CV_8UC1), averageMask;
	Mat zeros, ones, zerosMask, onesMask;

	// Calculate the average template
	threshold(acum, zeros, templates.size()*0.25, 1, THRESH_BINARY_INV);
	threshold(acum, ones, templates.size()*0.75, 1, THRESH_BINARY);
	averageTemplate.setTo(1, ones);
	averageTemplate.setTo(0, zeros);

	// Calculate the average mask
	threshold(acumMask, zerosMask, templates.size()*0.25, 1, THRESH_BINARY_INV);
	threshold(acumMask, onesMask, templates.size()*0.75, 1, THRESH_BINARY);
	bitwise_xor(zeros, ones, averageMask);				// EITHER 0 or 1 => consistent => mark as valid in mask
	averageMask.setTo(0, zerosMask);					// Disable the bits that are usually disabled in the mask

	return IrisTemplate(averageTemplate, averageMask, templates[0].encoderSignature);
}
