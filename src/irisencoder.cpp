#include <cmath>
#include <iostream>

#include "irisencoder.h"
#include "parameters.h"
#include "tools.h"

const double IrisEncoder::THETA0 = -M_PI/4.0;
const double IrisEncoder::THETA1 = (5.0/4.0) * M_PI;
const double IrisEncoder::RADIUS_TO_USE = 0.75;

IrisEncoder::IrisEncoder()
{
	this->normalizedTexture = NULL;
	this->normalizedNoiseMask = NULL;
}

IrisEncoder::~IrisEncoder()
{
}

IrisTemplate IrisEncoder::generateTemplate(const Mat& image, const SegmentationResult& segmentationResult)
{
	// We can only process grayscale images. If it's a color image, we need to convert it. Try to optimise whenever
	// possible.
	Mat_<uint8_t> bwimage;

	assert(image.depth() == CV_8U && (image.channels() == 1 || image.channels() == 3));

	if (image.channels() == 1) {
		bwimage = image;
	} else {
		cvtColor(image, bwimage, CV_BGR2GRAY);
	}

	Parameters* parameters = Parameters::getParameters();
	Size normalizedSize(parameters->normalizationWidth, parameters->normalizationHeight);

	this->normalizedTexture.create(normalizedSize);
	this->normalizedNoiseMask.create(normalizedSize);

	IrisEncoder::normalizeIris(bwimage, this->normalizedTexture, this->normalizedNoiseMask, segmentationResult, IrisEncoder::THETA0, IrisEncoder::THETA1, IrisEncoder::RADIUS_TO_USE);

	// Improve the iris mask
	this->extendMask();

	return this->encodeTexture(this->normalizedTexture, this->normalizedNoiseMask);
}

void IrisEncoder::extendMask()
{
	// Mask away pixels too far from the mean
	Scalar smean, sdev;
	meanStdDev(this->normalizedTexture, smean, sdev, this->normalizedNoiseMask);

	double mean = smean.val[0], dev = sdev.val[0];
	uint8_t uthresh = uint8_t(mean+dev);
	uint8_t lthresh = uint8_t(mean-dev);

	for (int y = 0; y < this->normalizedTexture.rows; y++) {
		uint8_t* row = this->normalizedTexture.ptr(y);
		for (int x = 0; x < this->normalizedTexture.cols; x++) {
			uint8_t val = row[x];
			if (val < lthresh || val > uthresh) {
				this->normalizedNoiseMask(y, x) = 0;
			}
		}
	}
}

void IrisEncoder::normalizeIris(const Mat_<uint8_t>& image, Mat_<uint8_t>& dest, Mat_<uint8_t>& destMask, const SegmentationResult& segmentationResult, double theta0, double theta1, double radius)
{
	int normalizedWidth = dest.cols, normalizedHeight = dest.rows;

	vector< pair<Point, Point> > irisPoints = Tools::iterateIris(segmentationResult,
		normalizedWidth, normalizedHeight, theta0, theta1, radius);

	// Initialize the mask to 1 (all bits enabled)
	destMask.setTo(Scalar(1));

	for (size_t i = 0; i < irisPoints.size(); i++) {
		Point imagePoint = irisPoints[i].second;
		Point coord = irisPoints[i].first;

		int ximage0 = int(floor(imagePoint.x));
		int ximage1 = int(ceil(imagePoint.x));
		int yimage0 = int(floor(imagePoint.y));
		int yimage1 = int(ceil(imagePoint.y));

		if (ximage0 < 0 || ximage1 >= image.cols || yimage0 < 0 || yimage1 >= image.rows) {
			dest(coord.y, coord.x) = 0;
			destMask(coord.y, coord.x) = 0;
		} else {
			double v1 = image(yimage0, ximage0);
			double v2 = image(yimage0, ximage1);
			double v3 = image(yimage1, ximage0);
			double v4 = image(yimage1, ximage1);
			dest(coord.y, coord.x) = (v1+v2+v3+v4)/4.0;
		}

		// See if (x,y) is occluded by an eyelid
		if (segmentationResult.eyelidsSegmented) {
			if (imagePoint.y <= segmentationResult.upperEyelid.value(imagePoint.x) || imagePoint.y >= segmentationResult.lowerEyelid.value(imagePoint.x)) {
				destMask(coord.y, coord.x) = 0;
			}
		}
	}
}

Size IrisEncoder::getOptimumTemplateSize(int width, int height)
{
	int optimumWidth = int(ceil(float(width)/32.0)) * 32; // Must be a multiple of 32
	return Size(optimumWidth, height);
}
