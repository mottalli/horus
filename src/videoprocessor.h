#pragma once

#include "common.h"
#include "qualitychecker.h"
#include "segmentator.h"
#include "loggaborencoder.h"

class VideoProcessorParameters
{
public:
	int bestFrameWaitCount;
	int focusThreshold;
	bool interlacedVideo;
	int correlationThreshold;
	float segmentationScoreThreshold;
	int minimumContourQuality;
	bool segmentEyelids;

	VideoProcessorParameters()
	{
		this->bestFrameWaitCount = 20;
		this->focusThreshold = 40;
		this->interlacedVideo = true;
		this->correlationThreshold = 92;
		this->segmentationScoreThreshold = 1.7;
		this->minimumContourQuality = 60;
		this->segmentEyelids = false;
	}
};

class VideoProcessor {
public:
	VideoProcessor();
	virtual ~VideoProcessor();
	
	typedef enum {
		UNPROCESSED,
		DEFOCUSED,
		INTERLACED,
		NO_EYE,
		FOCUSED_NO_IRIS,
		IRIS_LOW_QUALITY,
		IRIS_TOO_CLOSE,
		IRIS_TOO_FAR,
		FOCUSED_IRIS,
		GOT_TEMPLATE
	} VideoStatus;

	VideoProcessorParameters parameters;

	VideoStatus processFrame(const Mat& frame);

	void setWaitingFrames(int frames) { this->waitingFrames = frames; };

	QualityChecker qualityChecker;
	Segmentator segmentator;
	LogGaborEncoder irisEncoder;

	double lastFocusScore;
	QualityChecker::ValidationHeuristics lastIrisHeuristics;
	VideoStatus lastStatus;
	SegmentationResult lastSegmentationResult;
	double lastIrisQuality;
	
	IrisTemplate getTemplate() const;
	const Mat& getTemplateFrame() const { return this->templateFrame; }
	SegmentationResult getTemplateSegmentation() const { return this->templateSegmentation; }

	Mat lastFrame;

	Rect eyeROI;

private:
	unsigned int waitingFrames;

	VideoStatus doProcess(const Mat& frame);
	
	Mat templateFrame;
	SegmentationResult templateSegmentation;
	double templateIrisQuality;
	
	int templateWaitCount;
	int framesToSkip;
	bool waitingBestTemplate;

	CascadeClassifier eyeClassifier;
};

