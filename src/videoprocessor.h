#pragma once

#include "common.h"
#include "qualitychecker.h"
#include "segmentator.h"
#include "loggaborencoder.h"

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

	struct {
		int bestFrameWaitCount;
		int focusThreshold;
		bool interlacedVideo;
		int correlationThreshold;
		float segmentationScoreThreshold;
		int minimumContourQuality;
		bool segmentEyelids;
	} parameters;

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
	
	IrisTemplate getTemplate();
	const Mat& getTemplateFrame() const { return this->templateFrame; };
	SegmentationResult getTemplateSegmentation() const { return this->templateSegmentation; };

private:
	Mat lastFrame;

	unsigned int waitingFrames;

	VideoStatus doProcess(const Mat& frame);
	
	Mat templateFrame;
	SegmentationResult templateSegmentation;
	double templateIrisQuality;
	
	int templateWaitCount;
	int framesToSkip;
	bool waitingBestTemplate;
};

