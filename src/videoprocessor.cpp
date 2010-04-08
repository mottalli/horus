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
	this->lastStatus = DEFOCUSED;
	this->templateWaitCount = 0;
	this->templateIrisQuality = 0.0;
	this->waitingBestTemplate = false;
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
		if (this->templateWaitCount == BEST_FRAME_WAIT_COUNT) {
			// Got the iris template
			this->lastStatus = GOT_TEMPLATE;
			this->templateWaitCount = 0;
			this->waitingBestTemplate = false;
			this->templateIrisQuality = 0;
			
			// Wait before processing more images
			this->framesToSkip = 10;
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

	if (!qualityChecker.validateIris(image, this->lastSegmentationResult)) {
		// No iris found on the image, or the segmentation is incorrect
		return FOCUSED_NO_IRIS;
	}

	this->lastIrisQuality = qualityChecker.getIrisQuality(image, this->lastSegmentationResult);
	if (this->lastIrisQuality < parameters->minimumContourQuality) {
		// The image is kind of focused but the iris doesn't have enough quality
		/*float q = 0.2;

		if (this->lastSegmentationResult.irisCircle.radius*2 < parameters->expectedIrisDiameter*q) {
			// Iris too far?
			return IRIS_TOO_FAR;
		} else if (this->lastSegmentationResult.irisCircle.radius*2 > parameters->expectedIrisDiameter*q) {
			// Iris too close?
			return IRIS_TOO_CLOSE;
		} else {
			// Low quality for some reason...
			return IRIS_LOW_QUALITY;
		}*/
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
