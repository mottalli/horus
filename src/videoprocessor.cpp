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

	this->eyeClassifier.load("haarcascade_eye.xml");
	if (this->eyeClassifier.empty()) {
		throw runtime_error("Unable to load haarcascade_eye.xml classifier");
	}
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

	if (frame.channels() == 1) {
		this->lastFrameBW = this->lastFrame;
	} else {
		cvtColor(frame, this->lastFrameBW, CV_BGR2GRAY);
	}
	
	this->lastStatus = this->doProcess(frame);
	
	if (this->lastStatus == FOCUSED_IRIS) {
		this->waitingBestTemplate = true;
	}
	
	if (this->waitingBestTemplate) {
		if (this->lastStatus == FOCUSED_IRIS && this->lastIrisQuality > this->templateIrisQuality) {
			// Got a better quality image
			this->lastFrameBW.copyTo(this->templateFrame);
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
	const Mat& image = this->lastFrameBW;

	// Step 1 - See if there's an eye
	vector<Rect> classifications;

	Mat_<uint8_t> tmp;
	pyrDown(image, tmp);			//TODO: Move this to inner variable
	this->eyeClassifier.detectMultiScale(tmp, classifications, 2, 3, 0, Size(tmp.cols/4, tmp.rows/4));

	if (classifications.size() == 0) {
		return NO_EYE;
	}

	/*namedWindow("pyr");
	rectangle(tmp, classifications[0], CV_RGB(255,255,255));
	imshow("pyr", tmp);*/

	Rect& c = classifications[0];
	this->eyeROI = Rect(2*c.x, 2*c.y, 2*c.width, 2*c.height);



	// Step 2 - Check that the image is in focus
	/*this->lastFocusScore = this->qualityChecker.checkFocus(image(this->eyeROI));
	if (this->lastFocusScore < this->parameters.focusThreshold) {
		return DEFOCUSED;
	}*/

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

IrisTemplate VideoProcessor::getTemplate() const
{
	// const_cast is needed here before generateTemplate is not const... the other option is making irisEncoder as a mutable variable
	return const_cast<VideoProcessor*>(this)->irisEncoder.generateTemplate(this->templateFrame, this->templateSegmentation);
}
