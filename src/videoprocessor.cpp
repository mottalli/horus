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

VideoProcessor::VideoProcessor()
{
	this->lastStatus = DEFOCUSED;
	this->templateWaitCount = 0;
	this->templateIrisQuality = 0.0;
	this->waitingBestTemplate = false;
	this->waitingFrames = 40;
	this->framesToSkip = 0;
}

VideoProcessor::~VideoProcessor()
{
}

VideoProcessor::VideoStatus VideoProcessor::processFrame(const Mat& frame)
{
	if (this->framesToSkip > 0) {
		this->framesToSkip--;
		this->lastStatus = UNPROCESSED;
		return UNPROCESSED;
	}
	
	this->lastStatus = this->doProcess(frame);
	
	if (this->lastStatus == FOCUSED_IRIS) {
		this->waitingBestTemplate = true;
	}
	
	if (this->waitingBestTemplate) {
		if (this->lastStatus == FOCUSED_IRIS && this->lastIrisQuality > this->templateIrisQuality) {
			// Got a better quality image
			this->lastFrame.copyTo(this->templateFrame);
			this->templateIrisQuality = this->lastIrisQuality;
			this->templateSegmentation = this->lastSegmentationResult;
			this->templateWaitCount = 0;
		}
		
		this->templateWaitCount++;
		if (this->templateWaitCount >= this->parameters.bestFrameWaitCount) {
			// Got the iris template with the best quality
			this->lastStatus = GOT_TEMPLATE;
			this->templateWaitCount = 0;
			this->waitingBestTemplate = false;
			this->templateIrisQuality = 0;
			
			// Wait before processing more images
			this->framesToSkip = this->waitingFrames;
		}
	}
	
	return this->lastStatus;
}

VideoProcessor::VideoStatus VideoProcessor::doProcess(const Mat& frame)
{
	if (frame.channels() == 1) {
		frame.copyTo(this->lastFrame);
	} else {
		cvtColor(frame, this->lastFrame, CV_BGR2GRAY);
	}
	
	const Mat& image = this->lastFrame;

	this->lastFocusScore = this->qualityChecker.checkFocus(image);
	if (this->lastFocusScore < this->parameters.focusThreshold) {
		return DEFOCUSED;
	}

	if (this->parameters.interlacedVideo) {
		double interlacedCorrelation = this->qualityChecker.interlacedCorrelation(image);
		if (interlacedCorrelation < this->parameters.correlationThreshold) {
			return INTERLACED;
		}
	}

	this->lastSegmentationResult = segmentator.segmentImage(image);
	if (this->parameters.segmentEyelids) {
		segmentator.segmentEyelids(image, this->lastSegmentationResult);
	}

	this->lastIrisHeuristics = qualityChecker.validateIris(image, this->lastSegmentationResult);
	if (this->lastIrisHeuristics != QualityChecker::HAS_IRIS) {
		// No iris found on the image, or the segmentation is incorrect
		return FOCUSED_NO_IRIS;
	}

	this->lastIrisQuality = qualityChecker.getIrisQuality(image, this->lastSegmentationResult);
	if (this->lastIrisQuality < this->parameters.minimumContourQuality) {
		// The image is kind of focused but the iris doesn't have enough quality
		// TODO: Check whether to return IRIS_TOO_FAR or IRIS_TOO_CLOSE

		return IRIS_LOW_QUALITY;
	}

	// At this point we have a good quality image and we have enough reasons to believe
	// it's properly segmented.
	return FOCUSED_IRIS;
}

IrisTemplate VideoProcessor::getTemplate()
{
	return this->irisEncoder.generateTemplate(this->templateFrame, this->templateSegmentation);
}
