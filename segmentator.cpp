/*
 * File:   segmentator.cpp
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:37 PM
 */


#include <stdexcept>
#include <iostream>

#include "segmentator.h"
#include "helperfunctions.h"


Segmentator::Segmentator()
{
	this->buffers.workingImage = NULL;
}

Segmentator::~Segmentator() {
}

SegmentationResult Segmentator::segmentImage(const Image* image) {
	SegmentationResult result;
	const Image* imageToSegment;		// Can either user image or workingImage depending on image format

	this->setupBuffers(image);

	if (image->nChannels == 1 && image->depth == IPL_DEPTH_8U) {
		imageToSegment = image;
	} else {
		cvCvtColor(image, this->buffers.workingImage, CV_RGB2GRAY);
		imageToSegment = this->buffers.workingImage;
	}

	ContourAndCloseCircle pupilResult = this->_pupilSegmentator.segmentPupil(imageToSegment);
	ContourAndCloseCircle irisResult = this->_irisSegmentator.segmentIris(imageToSegment, pupilResult);

	result.pupilContour = pupilResult.first;
	result.pupilCircle = pupilResult.second;
	result.irisContour = irisResult.first;
	result.irisCircle = irisResult.second;

	std::pair<Parabola, Parabola> eyelids = this->_eyelidSegmentator.segmentEyelids(imageToSegment, result.pupilCircle, result.irisCircle);
	result.upperEyelid = eyelids.first;
	result.lowerEyelid = eyelids.second;

	return result;
};

void Segmentator::setupBuffers(const Image* image)
{
	if (this->buffers.workingImage == NULL || this->buffers.workingImage->width != image->width || this->buffers.workingImage->height != image->height) {
		if (this->buffers.workingImage != NULL) {
			cvReleaseImage(&this->buffers.workingImage);
		}

		this->buffers.workingImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	}

}
