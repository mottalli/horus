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
	
	
	this->initializeBuffers(tmpImage);
	IrisEncoder::normalizeIris(tmpImage, this->normalizedTexture, this->normalizedNoiseMask, segmentationResult);

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

void IrisEncoder::normalizeIris(const IplImage* image, IplImage* dest, CvMat* destMask, const SegmentationResult& segmentationResult)
{
	int normalizedWidth = dest->width, normalizedHeight = dest->height;

	std::vector< std::pair<CvPoint, CvPoint> > irisPoints = Tools::iterateIris(segmentationResult,
		normalizedWidth, normalizedHeight, IrisEncoder::THETA0, IrisEncoder::THETA1,
		IrisEncoder::RADIUS_TO_USE);

	// Initialize the mask to 1 (all bits enabled)
	cvSet(destMask, cvScalar(1,1,1));

	for (size_t i = 0; i < irisPoints.size(); i++) {
		CvPoint imagePoint = irisPoints[i].second;
		CvPoint coord = irisPoints[i].first;

		int ximage0 = int(std::floor(imagePoint.x));
		int ximage1 = int(std::ceil(imagePoint.x));
		int yimage0 = int(std::floor(imagePoint.y));
		int yimage1 = int(std::ceil(imagePoint.y));

		if (ximage0 < 0 || ximage1 >= image->width || yimage0 < 0 || yimage1 >= image->height) {
			cvSetReal2D(dest, coord.y, coord.x, 0);
			cvSetReal2D(destMask, coord.y, coord.x, 0);
		} else {
			double v1 = cvGetReal2D(image, yimage0, ximage0);
			double v2 = cvGetReal2D(image, yimage0, ximage1);
			double v3 = cvGetReal2D(image, yimage1, ximage0);
			double v4 = cvGetReal2D(image, yimage1, ximage1);
			cvSetReal2D(dest, coord.y, coord.x, (v1+v2+v3+v4)/4.0);
		}

		// See if (x,y) is occluded by an eyelid
		if (segmentationResult.eyelidsSegmented) {
			if (imagePoint.y <= segmentationResult.upperEyelid.value(imagePoint.x) || imagePoint.y >= segmentationResult.lowerEyelid.value(imagePoint.x)) {
				cvSetReal2D(destMask, coord.y, coord.x, 0);
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
