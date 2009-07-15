/*
 * videoprocessor.cpp
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#include "videoprocessor.h"
#include <stdexcept>
#include <iostream>
#include <cmath>

void cvShiftDFT(CvArr * src_arr, CvArr * dst_arr );

VideoProcessor::VideoProcessor()
{
	this->buffers.lastFrame = NULL;
	this->lastStatus = DEFOCUSED;
}

VideoProcessor::~VideoProcessor()
{
	if (this->buffers.lastFrame != NULL) {
		cvReleaseImage(&this->buffers.lastFrame);
	}
}

VideoProcessor::VideoStatus VideoProcessor::processFrame(const Image* frame)
{
	this->initializeBuffers(frame);
	this->lastStatus = this->doProcess(frame);
	return this->lastStatus;
}

VideoProcessor::VideoStatus VideoProcessor::doProcess(const Image* frame)
{
	const Image* image;
	if (frame->nChannels == 1) {
		//cvCopy(frame, this->buffers.lastFrame);
		image = frame;
	} else {
		cvCvtColor(frame, this->buffers.lastFrame, CV_BGR2GRAY);
		image = this->buffers.lastFrame;
	}

	Parameters* parameters = Parameters::getParameters();

	this->lastFocusScore = this->qualityChecker.checkFocus(image);
	if (this->lastFocusScore < parameters->focusThreshold) {
		return DEFOCUSED;
	}

	if (parameters->interlacedVideo) {
		double interlacedCorrelation = this->qualityChecker.interlacedCorrelation(image);
		if (interlacedCorrelation < parameters->correlationThreshold) {
			return INTERLACED;
		}
	}

	this->lastSegmentationResult = segmentator.segmentImage(image);
	if (parameters->segmentEyelids) {
		segmentator.segmentEyelids(image, this->lastSegmentationResult);
	}

	this->lastSegmentationScore = qualityChecker.segmentationScore(image, this->lastSegmentationResult);
	if (this->lastSegmentationScore < parameters->segmentationScoreThreshold) {
		// No iris found on the image, or the segmentation is incorrect
		return FOCUSED_NO_IRIS;
	}

	if (!qualityChecker.checkIrisQuality(image, this->lastSegmentationResult)) {
		// The image is kind of focused but the iris doesn't have enough quality
		float q = 0.2;

		if (this->lastSegmentationResult.irisCircle.radius*2 < parameters->expectedIrisDiameter*q) {
			// Iris too far?
			return IRIS_TOO_FAR;
		} else if (this->lastSegmentationResult.irisCircle.radius*2 > parameters->expectedIrisDiameter*q) {
			// Iris too close?
			return IRIS_TOO_CLOSE;
		} else {
			// Low quality for some reason...
			return IRIS_LOW_QUALITY;
		}
	}

	// At this point we have a good quality image and we have enough reasons to believe
	// it's properly segmented.
	if (false) {
		//TODO: wait for the "best" image
		return FOCUSED_IRIS;
	} else {
		// Got a good iris image
		cvCopy(image, this->buffers.bestFrame);
		return GOT_TEMPLATE;
	}
}

IrisTemplate VideoProcessor::getTemplate()
{
	if (this->lastStatus != GOT_TEMPLATE) {
		throw std::runtime_error("Requested iris template but no iris detected in image");
	}

	return this->irisEncoder.generateTemplate(this->buffers.bestFrame, this->lastSegmentationResult);
}

void VideoProcessor::initializeBuffers(const Image* frame)
{
	if (this->buffers.lastFrame == NULL || !SAME_SIZE(this->buffers.lastFrame, frame)) {
		if (this->buffers.lastFrame != NULL) {
			cvReleaseImage(&this->buffers.lastFrame);
			cvReleaseImage(&this->buffers.bestFrame);
		}
		this->buffers.lastFrame = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
		this->buffers.bestFrame = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
	}
}
