/*
 * irisencoder.cpp
 *
 *  Created on: Jun 10, 2009
 *      Author: marcelo
 */

#include <cmath>
#include <iostream>

#include "irisencoder.h"
#include "parameters.h"
#include "tools.h"

IrisEncoder::IrisEncoder()
{
	this->normalizedTexture = NULL;
	this->normalizedNoiseMask = NULL;
	this->resizedTexture = NULL;
	this->resizedNoiseMask = NULL;
}

IrisEncoder::~IrisEncoder()
{
}

IrisTemplate IrisEncoder::generateTemplate(const IplImage* image, const SegmentationResult& segmentationResult)
{
	assert(image->nChannels == 1);
	this->initializeBuffers(image);
	IrisEncoder::normalizeIris(image, this->normalizedTexture, this->normalizedNoiseMask, segmentationResult);

	// Improve the iris mask
	this->extendMask();

	if (SAME_SIZE(this->normalizedTexture, this->resizedTexture)) {
		// No resizing needs to be done -- process the texture directly
		return this->encodeTexture(this->normalizedTexture, this->normalizedNoiseMask);
	} else {
		// The texture needs to be resized
		cvResize(this->normalizedTexture, this->resizedTexture, CV_INTER_CUBIC);
		cvResize(this->normalizedNoiseMask, this->resizedNoiseMask, CV_INTER_NN);		// Needs to be NN so it keeps the right bits
		return this->encodeTexture(this->resizedTexture, this->resizedNoiseMask);
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

void IrisEncoder::normalizeIris(const IplImage* image, IplImage* dest, CvMat* destMask, const SegmentationResult& segmentationResult)
{
	int normalizedWidth = dest->width, normalizedHeight = dest->height;
	const Contour& pupilContour = segmentationResult.pupilContour;
	const Contour& irisContour = segmentationResult.irisContour;
	CvPoint p0, p1;

	// Initialize the mask to 1 (all bits enabled)
	cvSet(destMask, cvScalar(1,1,1));

	// We want to exclude the upper quarter.
	double theta0 = IrisEncoder::THETA0;
	double theta1 = IrisEncoder::THETA1;
	double radiusToUse = IrisEncoder::RADIUS_TO_USE;		// Only use three-quarters of the radius

	for (int x = 0; x < normalizedWidth; x++) {
		double theta = (double(x)/double(normalizedWidth)) * (theta1-theta0) + theta0;
		if (theta < 0) theta = 2.0 * M_PI + theta;

		// Remember we're mapping pupilContour[0] to 0 degrees and pupilContour[size-1] to "almost" 360 degrees
		assert(theta >= 0 && theta <= 2.0*M_PI);
		double w = (theta/(2.0*M_PI))*double(pupilContour.size());

		p0 = pupilContour[int(std::floor(w))];
		p1 = pupilContour[int(std::ceil(w)) % pupilContour.size()];
		double prop = w-std::floor(w);
		double xfrom = double(p0.x) + double(p1.x-p0.x)*prop;
		double yfrom = double(p0.y) + double(p1.y-p0.y)*prop;

		w = (theta/(2.0*M_PI))*double(irisContour.size());
		p0 = irisContour[int(std::floor(w))];
		p1 = irisContour[int(std::ceil(w)) % irisContour.size()];
		prop = w-std::floor(w);
		double xto = double(p0.x) + double(p1.x-p0.x)*prop;
		double yto = double(p0.y) + double(p1.y-p0.y)*prop;

		for (int y = 0; y < normalizedHeight; y++) {
			w = (double(y)/double(normalizedHeight-1)) * radiusToUse;
			double ximage = xfrom + w*(xto-xfrom);
			double yimage = yfrom + w*(yto-yfrom);

			int ximage0 = int(std::floor(ximage));
			int ximage1 = int(std::ceil(ximage));
			int yimage0 = int(std::floor(yimage));
			int yimage1 = int(std::ceil(yimage));

			if (ximage0 < 0 || ximage1 >= image->width || yimage0 < 0 || yimage1 >= image->height) {
				cvSetReal2D(dest, y, x, 0);
				cvSetReal2D(destMask, y, x, 0);
			} else {
				double v1 = cvGetReal2D(image, yimage0, ximage0);
				double v2 = cvGetReal2D(image, yimage0, ximage1);
				double v3 = cvGetReal2D(image, yimage1, ximage0);
				double v4 = cvGetReal2D(image, yimage1, ximage1);
				cvSetReal2D(dest, y, x, (v1+v2+v3+v4)/4.0);
			}

			// See if (x,y) is occluded by an eyelid
			if (segmentationResult.eyelidsSegmented) {
				if (yimage <= segmentationResult.upperEyelid.value(ximage) || yimage >= segmentationResult.lowerEyelid.value(ximage)) {
					cvSetReal2D(destMask, y, x, 0);
				}
			}
		}
	}

	cvDilate(destMask, destMask);
}

void IrisEncoder::initializeBuffers(const IplImage* image)
{
	Parameters* parameters = Parameters::getParameters();

	if (this->normalizedTexture == NULL || this->normalizedTexture->width != parameters->normalizationWidth || this->normalizedTexture->height != parameters->normalizationHeight) {
		//TODO: release if they were already created
		this->normalizedTexture = cvCreateImage(cvSize(parameters->normalizationWidth,parameters->normalizationHeight), IPL_DEPTH_8U, 1);
		this->normalizedNoiseMask = cvCreateMat(parameters->normalizationHeight,parameters->normalizationWidth, CV_8U);
	}

	if (this->resizedTexture == NULL || this->resizedTexture->width != parameters->templateWidth || this->resizedTexture->height != parameters->templateHeight) {
		this->resizedTexture = cvCreateImage(cvSize(parameters->templateWidth,parameters->templateHeight), IPL_DEPTH_8U, 1);
		this->resizedNoiseMask = cvCreateMat(parameters->templateHeight, parameters->templateWidth, CV_8U);
	}
}
