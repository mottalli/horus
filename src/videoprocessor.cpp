/*
 * videoprocessor.cpp
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#include <stdexcept>
#include <iostream>
#include <cmath>

#include "videoprocessor.h"
#include "tools.h"

VideoProcessor::VideoProcessor()
{
	this->lastStatus = DEFOCUSED;
	this->templateWaitCount = 0;
	this->templateIrisQuality = 0.0;
	this->waitingTemplate = false;
	this->framesToSkip = 0;
}

VideoProcessor::~VideoProcessor()
{
}

VideoProcessor::VideoStatus VideoProcessor::processFrame(const Mat& frame)
{
	frame.copyTo(this->lastFrame);

	if (this->framesToSkip > 0) {
		this->framesToSkip--;
		this->lastStatus = UNPROCESSED;
		return UNPROCESSED;
	}

	Tools::toGrayscale(this->lastFrame, this->lastFrameBW, false);
	
	this->lastStatus = this->doProcess(this->lastFrameBW);
	
	if (this->lastStatus == FOCUSED_IRIS && !this->waitingTemplate) {		// A new iris has been detected. Start the template generation process
		this->resetCapture();
		this->waitingTemplate = true;
	}

	if (this->waitingTemplate) {
		this->templateWaitCount++;

		if (this->lastStatus == FOCUSED_IRIS) {
			CapturedTemplate capturedTemplate;
			capturedTemplate.image = this->lastFrameBW.clone();
			capturedTemplate.irisTemplate = this->irisEncoder.generateTemplate(this->lastFrameBW, this->lastSegmentationResult);
			capturedTemplate.quality = this->lastIrisQuality;
			capturedTemplate.segmentationResult = this->lastSegmentationResult;

			this->templateBuffer.push_back(capturedTemplate);

			if (this->templateBuffer.size() == 1) {
				this->bestTemplateIdx = 0;
			} else if (this->lastIrisQuality > this->templateBuffer[this->bestTemplateIdx].quality) {		// Got a better quality image
				this->bestTemplateIdx = templateBuffer.size()-1;
			}

			this->lastTemplate = capturedTemplate.irisTemplate;

			if (this->templateBuffer.size() < this->parameters.minCountForTemplateAveraging) {
				this->templateWaitCount = 0;			// Reset the wait count (still capturing)
			} else {
				this->lastStatus = FINISHED_CAPTURE;
			}
		}

		if (this->templateWaitCount >= this->parameters.templateWaitTimeout) {		// Finished the burst
			if (this->templateBuffer.size() >= this->parameters.minCountForTemplateAveraging) {
				// Enough templates - Can calculate average!
				// Check if we have a "good" template
				IrisTemplate irisTemplate = this->getAverageTemplate();
				if (this->qualityChecker.irisTemplateQuality(irisTemplate) < this->parameters.minAverageTemplateQuality) {
					// Bad template - try another one
					this->resetCapture();
				} else {
					this->lastStatus = GOT_TEMPLATE;
					this->waitingTemplate = false;
					this->templateWaitCount = 0;
					// Wait before processing more images (if enabled)
					this->framesToSkip = (this->parameters.pauseAfterCapture ? this->parameters.pauseFrames : 0);
				}
			} else {
				this->resetCapture();		// Didn't get enough images -- restart the capture process
			}
		}
	}
	
	return this->lastStatus;
}

VideoProcessor::VideoStatus VideoProcessor::doProcess(const GrayscaleImage& image)
{
	// Step 1 - Check that the image is in focus
	this->lastFocusScore = this->qualityChecker.checkFocus(image);
	if (this->lastFocusScore < this->parameters.focusThreshold) {
		return DEFOCUSED;
	}

	if (this->parameters.doEyeDetect) {
		// Step 2 - See if there's an eye
		if (!this->eyeDetect.detectEye(image)) {
			this->eyeROI = Rect();
			return NO_EYE;
		}

		// "Smooth" the ROI across frames
		Rect roi = this->eyeDetect.eyeRect;
		int x = roi.x;
		int y = roi.y;
		int w = roi.width;
		int h = roi.height;

		if (this->eyeROI.width > 0) {
			x = (x+this->eyeROI.x) / 2;
			y = (y+this->eyeROI.y) / 2;
			w = (w+this->eyeROI.width) / 2;
			h = (h+this->eyeROI.height) / 2;
		}
		this->eyeROI = Rect(x, y, w, h);
	} else {
		this->eyeROI = Rect(0,0,image.cols, image.rows);
	}

	if (this->parameters.interlacedVideo) {
		double interlacedCorrelation = this->qualityChecker.interlacedCorrelation(image);
		if (interlacedCorrelation < this->parameters.correlationThreshold) {
			return INTERLACED;
		}
	}

	segmentator.setEyeROI(this->eyeROI);
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

IrisTemplate VideoProcessor::getBestTemplate() const
{
	return this->templateBuffer[this->bestTemplateIdx].irisTemplate;
}

IrisTemplate VideoProcessor::getAverageTemplate() const
{
	vector<const IrisTemplate*> templates;
	for (vector<CapturedTemplate>::const_iterator it = this->templateBuffer.begin(); it != this->templateBuffer.end(); it++) {
		templates.push_back( &((*it).irisTemplate) );
	}
	return IrisEncoder::averageTemplates(templates);
}

GrayscaleImage VideoProcessor::getBestTemplateFrame() const
{
	GrayscaleImage res;
	Tools::stretchHistogram(this->templateBuffer[this->bestTemplateIdx].image, res, 0.01, 0);
	return res;
}

const SegmentationResult& VideoProcessor::getBestTemplateSegmentation() const
{
	return this->templateBuffer[this->bestTemplateIdx].segmentationResult;
}
