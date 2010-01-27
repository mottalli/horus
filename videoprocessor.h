/*
 * videoprocessor.h
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#pragma once

#include "common.h"
#include "qualitychecker.h"
#include "segmentator.h"
#include "irisencoder.h"

class VideoProcessor {
public:
	VideoProcessor();
	virtual ~VideoProcessor();

	struct {
		Image* lastFrame;
		Image* bestFrame;
	} buffers;

	typedef enum {
		DEFOCUSED,
		INTERLACED,
		FOCUSED_NO_IRIS,
		IRIS_LOW_QUALITY,
		IRIS_TOO_CLOSE,
		IRIS_TOO_FAR,
		FOCUSED_IRIS,
		GOT_TEMPLATE
	} VideoStatus;

	VideoStatus processFrame(const Image* frame);
	IrisTemplate getTemplate();

	QualityChecker qualityChecker;
	Segmentator segmentator;
	IrisEncoder irisEncoder;

	VideoStatus lastStatus;
	SegmentationResult lastSegmentationResult;
	double lastSegmentationScore;
	double lastFocusScore;


private:
	VideoStatus doProcess(const Image* frame);
	void initializeBuffers(const Image* frame);
};

