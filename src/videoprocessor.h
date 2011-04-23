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
		FOCUSED_NO_IRIS,
		IRIS_LOW_QUALITY,
		IRIS_TOO_CLOSE,
		IRIS_TOO_FAR,
		FOCUSED_IRIS,
		GOT_TEMPLATE
	} VideoStatus;

	typedef struct {
		Mat_<uint8_t> image;
		SegmentationResult segmentationResult;
		IrisTemplate irisTemplate;
		double quality;
	} CapturedTemplate;


	VideoProcessorParameters parameters;

	VideoStatus processFrame(const Mat& frame);

	void setWaitingFrames(int frames) { this->waitingFrames = frames; }

	QualityChecker qualityChecker;
	Segmentator segmentator;
	LogGaborEncoder irisEncoder;

	double lastFocusScore;
	QualityChecker::ValidationHeuristics lastIrisHeuristics;
	VideoStatus lastStatus;
	SegmentationResult lastSegmentationResult;
	double lastIrisQuality;
	IrisTemplate lastTemplate;
	
	IrisTemplate getTemplate() const;
	const Mat& getTemplateFrame() const;
	SegmentationResult getTemplateSegmentation() const;

	Mat lastFrame;

private:
	Mat_<uint8_t> lastFrameBW;
	unsigned int waitingFrames;

	VideoStatus doProcess(const Mat& frame);
	
	int templateWaitCount;
	int framesToSkip;
	bool waitingBestTemplate;
	size_t bestTemplateIdx;

	vector<CapturedTemplate> templateBuffer;
};

