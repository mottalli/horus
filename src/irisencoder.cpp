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

IrisTemplate IrisEncoder::generateTemplate(const IplImage* image, const SegmentationResult& segmentationResult)
{
	// We can only process grayscale images. If it's a color image, we need to convert it. Try to optimise whenever
	// possible.
	IplImage* tmpImage;
	if (image->nChannels == 1) {
		tmpImage = const_cast<IplImage*>(image);			// const_cast is needed because tmpImage cannot be defined as const, but we're sure
															// we're not going to modify the image
	} else {
		tmpImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
		cvCvtColor(image, tmpImage, CV_BGR2GRAY);
	}
	
	
	//this->initializeBuffers(tmpImage);

	Parameters* parameters = Parameters::getParameters();
	CvSize normalizedSize = cvSize(parameters->normalizationWidth, parameters->normalizationHeight);
	Tools::updateSize(&this->normalizedTexture, normalizedSize);
	Tools::updateSize(&this->normalizedNoiseMask, normalizedSize);

	IrisEncoder::normalizeIris(tmpImage, this->normalizedTexture, this->normalizedNoiseMask, segmentationResult, IrisEncoder::THETA0, IrisEncoder::THETA1, IrisEncoder::RADIUS_TO_USE);

	// Improve the iris mask
	this->extendMask();

	return this->encodeTexture(this->normalizedTexture, this->normalizedNoiseMask);
	
	if (tmpImage != image) {
		cvReleaseImage(&tmpImage);
	}
}

void IrisEncoder::extendMask()
{
	// Mask away pixels too far from the mean
	CvScalar smean, sdev;
	cvAvgSdv(this->normalizedTexture, &smean, &sdev, this->normalizedNoiseMask);
	double mean = smean.val[0], dev = sdev.val[0];
	uint8_t uthresh = uint8_t(mean+dev);
	uint8_t lthresh = uint8_t(mean-dev);

	for (int y = 0; y < this->normalizedTexture->height; y++) {
		uint8_t* row = ((uint8_t*)this->normalizedTexture->imageData) + y*this->normalizedTexture->widthStep;
		for (int x = 0; x < this->normalizedTexture->width; x++) {
			uint8_t val = row[x];
			if (val < lthresh || val > uthresh) {
				cvSetReal2D(this->normalizedNoiseMask, y, x, 0);
			}
		}
	}
}

void IrisEncoder::normalizeIris(const IplImage* image, IplImage* dest, CvMat* destMask, const SegmentationResult& segmentationResult, double theta0, double theta1, double radius)
{
	int normalizedWidth = dest->width, normalizedHeight = dest->height;

	vector< pair<Point, Point> > irisPoints = Tools::iterateIris(segmentationResult,
		normalizedWidth, normalizedHeight, theta0, theta1, radius);

	// Initialize the mask to 1 (all bits enabled)
	if (destMask) {
		cvSet(destMask, cvScalar(1));
	}

	for (size_t i = 0; i < irisPoints.size(); i++) {
		Point imagePoint = irisPoints[i].second;
		Point coord = irisPoints[i].first;

		int ximage0 = int(floor(imagePoint.x));
		int ximage1 = int(ceil(imagePoint.x));
		int yimage0 = int(floor(imagePoint.y));
		int yimage1 = int(ceil(imagePoint.y));

		if (ximage0 < 0 || ximage1 >= image->width || yimage0 < 0 || yimage1 >= image->height) {
			cvSetReal2D(dest, coord.y, coord.x, 0);
			if (destMask) {
				cvSetReal2D(destMask, coord.y, coord.x, 0);
			}
		} else {
			double v1 = cvGetReal2D(image, yimage0, ximage0);
			double v2 = cvGetReal2D(image, yimage0, ximage1);
			double v3 = cvGetReal2D(image, yimage1, ximage0);
			double v4 = cvGetReal2D(image, yimage1, ximage1);
			cvSetReal2D(dest, coord.y, coord.x, (v1+v2+v3+v4)/4.0);
		}

		// See if (x,y) is occluded by an eyelid
		if (destMask && segmentationResult.eyelidsSegmented) {
			if (imagePoint.y <= segmentationResult.upperEyelid.value(imagePoint.x) || imagePoint.y >= segmentationResult.lowerEyelid.value(imagePoint.x)) {
				cvSetReal2D(destMask, coord.y, coord.x, 0);
			}
		}
	}

	if (destMask) {
		cvDilate(destMask, destMask);
	}
}

CvSize IrisEncoder::getOptimumTemplateSize(int width, int height)
{
	int optimumWidth = int(ceil(float(width)/32.0)) * 32; // Must be a multiple of 32
	return cvSize(optimumWidth, height);
}
