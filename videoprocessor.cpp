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
	this->lastStatus = this->doProcess(frame);
	return this->lastStatus;
}

VideoProcessor::VideoStatus VideoProcessor::doProcess(const Image* frame)
{
	if (frame->nChannels == 1) {
		cvCopy(frame, this->buffers.lastFrame);
	} else {
		cvCvtColor(frame, this->buffers.lastFrame, CV_BGR2GRAY);
	}

	Parameters* parameters = Parameters::getParameters();

	double focus = this->qualityChecker.checkFocus(this->buffers.lastFrame);
	if (focus < parameters->focusThreshold) {
		return DEFOCUSED;
	}

	if (parameters->interlacedVideo) {
		double interlacedCorrelation = this->qualityChecker.interlacedCorrelation(this->buffers.lastFrame);
		if (interlacedCorrelation < parameters->correlationThreshold) {
			return DEFOCUSED;

		}
	}

	SegmentationResult sr = segmentator.segmentImage(this->buffers.lastFrame);
	/*if (!qualityChecker.segmentationScore(this->buffers.lastFrame, sr)) {
		// No iris found on the image, or the segmentation is incorrect
		return FOCUSED_NO_IRIS;
	}*/

	if (!qualityChecker.checkIrisQuality(this->buffers.lastFrame, sr)) {
		// The image is kind of focused but the iris doesn't have enough quality
		float q = 0.2;

		if (sr.irisCircle.radius*2 < parameters->expectedIrisDiameter*q) {
			// Iris too far?
			return IRIS_TOO_FAR;
		} else if (sr.irisCircle.radius*2 > parameters->expectedIrisDiameter*q) {
			// Iris too close?
			return IRIS_TOO_CLOSE;
		} else {
			// Low quality for some reason...
			return IRIS_LOW_QUALITY;
		}
	}

	// At this point we have a good quality image and we have enough reasons to believe
	// it's properly segmented.
	if (true) {
		return FOCUSED_IRIS;
	} else {
		//TODO
		return GOT_TEMPLATE;
	}
}

IrisTemplate VideoProcessor::getTemplate()
{
	if (this->lastStatus != FOCUSED_IRIS || this->lastStatus != GOT_TEMPLATE) {
		throw std::runtime_error("Requested iris template but no iris detected in image");
	}

	//TODO
	return IrisTemplate();
}

void VideoProcessor::initializeBuffers(const Image* frame)
{
	if (this->buffers.lastFrame == NULL || !SAME_SIZE(this->buffers.lastFrame, frame)) {
		if (this->buffers.lastFrame != NULL) {
			cvReleaseImage(&this->buffers.lastFrame);
		}
		this->buffers.lastFrame = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
	}
}
