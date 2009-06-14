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

IrisEncoder::IrisEncoder() :
	filter(1.0/32.0, 0.5)
{
	this->buffers.noiseMask = NULL;
	this->buffers.normalizedTexture = NULL;
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
	IrisEncoder::normalizeIris(image, this->buffers.normalizedTexture, this->buffers.noiseMask, segmentationResult);

	this->applyFilter();

	return IrisTemplate(this->buffers.thresholdedTexture, this->buffers.noiseMask);
}

void IrisEncoder::applyFilter()
{
	this->filter.applyFilter(this->buffers.normalizedTexture, this->buffers.filteredTexture);
	cvSplit(this->buffers.filteredTexture, this->buffers.filteredTextureReal, this->buffers.filteredTextureImag, NULL, NULL);
	cvThreshold(this->buffers.filteredTextureImag, this->buffers.thresholdedTexture, 0, 1, CV_THRESH_BINARY);
}

void IrisEncoder::normalizeIris(const Image* image, Image* dest, CvMat* destMask, const SegmentationResult& segmentationResult)
{
	int normalizedWidth = dest->width, normalizedHeight = dest->height;
	const Contour& pupilContour = segmentationResult.pupilContour;
	const Contour& irisContour = segmentationResult.irisContour;
	CvPoint p0, p1;

	//assert(std::abs(double(pupilContour[0].y - irisContour[0].y)) < 2);		// Both contours must be aligned

	cvSet(destMask, cvScalar(1,1,1));
	for (int x = 0; x < normalizedWidth; x++) {
		double q = double(x)/double(normalizedWidth);
		double theta = q*2.0*M_PI;

		double w = q*double(pupilContour.size());
		p0 = pupilContour[int(std::floor(w))];
		p1 = pupilContour[int(std::ceil(w))];
		double prop = w-std::floor(w);
		double xfrom = double(p0.x) + double(p1.x-p0.x)*prop;
		double yfrom = double(p0.y) + double(p1.y-p0.y)*prop;

		w = q*double(irisContour.size());
		p0 = irisContour[int(std::floor(w))];
		p1 = irisContour[int(std::ceil(w))];
		prop = w-std::floor(w);
		double xto = double(p0.x) + double(p1.x-p0.x)*prop;
		double yto = double(p0.y) + double(p1.y-p0.y)*prop;

		for (int y = 0; y < normalizedHeight; y++) {
			w = double(y)/double(normalizedHeight-1);
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
}

void IrisEncoder::initializeBuffers(const Image* image)
{
	Parameters* parameters = Parameters::getParameters();

	if (this->buffers.normalizedTexture == NULL || this->buffers.normalizedTexture->width != parameters->templateWidth || this->buffers.normalizedTexture->height != parameters->templateHeight) {
		if (this->buffers.normalizedTexture != NULL) {
			cvReleaseImage(&this->buffers.normalizedTexture);
			cvReleaseImage(&this->buffers.filteredTexture);
		}
		this->buffers.normalizedTexture = cvCreateImage(cvSize(parameters->templateWidth,parameters->templateHeight), IPL_DEPTH_8U, 1);
		this->buffers.filteredTexture = cvCreateImage(cvSize(parameters->templateWidth,parameters->templateHeight), IPL_DEPTH_32F, 2);
		this->buffers.filteredTextureReal = cvCreateImage(cvSize(parameters->templateWidth,parameters->templateHeight), IPL_DEPTH_32F, 1);
		this->buffers.filteredTextureImag = cvCreateImage(cvSize(parameters->templateWidth,parameters->templateHeight), IPL_DEPTH_32F, 1);
		this->buffers.noiseMask = cvCreateMat(parameters->templateHeight, parameters->templateWidth, CV_8U);
		this->buffers.thresholdedTexture = cvCreateMat(parameters->templateHeight, parameters->templateWidth, CV_8U);
	}
}
