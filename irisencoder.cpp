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

IrisEncoder::IrisEncoder() :
	filter(1.0/40.0, 0.5)
{
	this->buffers.normalizedTexture = NULL;
	this->buffers.resizedTexture = NULL;
	this->buffers.normalizedNoiseMask = NULL;
	this->buffers.resizedNoiseMask = NULL;
	this->buffers.thresholdedTexture = NULL;
	this->buffers.filteredTexture = NULL;
	this->buffers.filteredTextureReal = NULL;
	this->buffers.filteredTextureImag = NULL;
}

IrisEncoder::~IrisEncoder()
{
}

IrisTemplate IrisEncoder::generateTemplate(const Image* image, const SegmentationResult& segmentationResult)
{
	assert(image->nChannels == 1);
	this->initializeBuffers(image);
	IrisEncoder::normalizeIris(image, this->buffers.normalizedTexture, this->buffers.normalizedNoiseMask, segmentationResult);

	// Mask away pixels too far from the mean
	CvScalar smean, sdev;
	cvAvgSdv(this->buffers.normalizedTexture, &smean, &sdev, this->buffers.normalizedNoiseMask);
	double mean = smean.val[0], dev = sdev.val[0];
	uint8_t uthresh = uint8_t(mean+dev);
	uint8_t lthresh = uint8_t(mean-dev);

	for (int y = 0; y < this->buffers.normalizedTexture->height; y++) {
		uint8_t* row = ((uint8_t*)this->buffers.normalizedTexture->imageData) + y*this->buffers.normalizedTexture->widthStep;
		for (int x = 0; x < this->buffers.normalizedTexture->width; x++) {
			uint8_t val = row[x];
			if (val < lthresh || val > uthresh) {
				cvSetReal2D(this->buffers.normalizedNoiseMask, y, x, 0);
			}
		}
	}

	this->applyFilter();

	return IrisTemplate(this->buffers.thresholdedTexture, this->buffers.resizedNoiseMask);
}

void IrisEncoder::applyFilter()
{
	// Must resize the normalized texture to the size of the template
	if (SAME_SIZE(this->buffers.normalizedTexture, this->buffers.resizedTexture)) {
		cvCopy(this->buffers.normalizedTexture, this->buffers.resizedTexture);
		cvCopy(this->buffers.normalizedNoiseMask, this->buffers.resizedNoiseMask);
	} else {
		cvResize(this->buffers.normalizedTexture, this->buffers.resizedTexture);
		cvResize(this->buffers.normalizedNoiseMask, this->buffers.resizedNoiseMask);
	}

	this->filter.applyFilter(this->buffers.resizedTexture, this->buffers.filteredTexture);
	cvSplit(this->buffers.filteredTexture, this->buffers.filteredTextureReal, this->buffers.filteredTextureImag, NULL, NULL);
	cvThreshold(this->buffers.filteredTextureReal, this->buffers.thresholdedTexture, 0, 1, CV_THRESH_BINARY);
}

void IrisEncoder::normalizeIris(const Image* image, Image* dest, CvMat* destMask, const SegmentationResult& segmentationResult)
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

void IrisEncoder::initializeBuffers(const Image* image)
{
	Parameters* parameters = Parameters::getParameters();

	if (this->buffers.normalizedTexture == NULL || this->buffers.normalizedTexture->width != parameters->normalizationWidth || this->buffers.normalizedTexture->height != parameters->normalizationHeight) {
		//TODO: release if they were already created
		this->buffers.normalizedTexture = cvCreateImage(cvSize(parameters->normalizationWidth,parameters->normalizationHeight), IPL_DEPTH_8U, 1);
		this->buffers.normalizedNoiseMask = cvCreateMat(parameters->normalizationHeight,parameters->normalizationWidth, CV_8U);
	}

	if (this->buffers.resizedTexture == NULL || this->buffers.resizedTexture->width != parameters->templateWidth || this->buffers.resizedTexture->height != parameters->templateHeight) {
		this->buffers.resizedTexture = cvCreateImage(cvSize(parameters->templateWidth,parameters->templateHeight), IPL_DEPTH_8U, 1);
		this->buffers.resizedNoiseMask = cvCreateMat(parameters->templateHeight, parameters->templateWidth, CV_8U);
		this->buffers.filteredTexture = cvCreateImage(cvSize(parameters->templateWidth,parameters->templateHeight), IPL_DEPTH_32F, 2);
		this->buffers.filteredTextureReal = cvCreateImage(cvSize(parameters->templateWidth,parameters->templateHeight), IPL_DEPTH_32F, 1);
		this->buffers.filteredTextureImag = cvCreateImage(cvSize(parameters->templateWidth,parameters->templateHeight), IPL_DEPTH_32F, 1);
		this->buffers.thresholdedTexture = cvCreateMat(parameters->templateHeight, parameters->templateWidth, CV_8U);
	}
}
