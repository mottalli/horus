#pragma once

#include "common.h"
#include "qualitychecker.h"
#include "segmentator.h"
#include "loggaborencoder.h"

class VideoProcessor {
public:
	VideoProcessor();
	virtual ~VideoProcessor();
	
	static const unsigned int BEST_FRAME_WAIT_COUNT = 5;

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

	VideoStatus processFrame(const Mat& frame);

	QualityChecker qualityChecker;
	Segmentator segmentator;
	LogGaborEncoder irisEncoder;

	double lastFocusScore;
	VideoStatus lastStatus;
	SegmentationResult lastSegmentationResult;
	double lastIrisQuality;
	
	IrisTemplate getTemplate();
	const Mat& getTemplateFrame() const { return this->templateFrame; };
	SegmentationResult getTemplateSegmentation() const { return this->templateSegmentation; };

private:
	Mat lastFrame;

	VideoStatus doProcess(const Mat& frame);
	
	Mat templateFrame;
	SegmentationResult templateSegmentation;
	double templateIrisQuality;
	
	unsigned int templateWaitCount;
	unsigned int framesToSkip;
	bool waitingBestTemplate;
};

