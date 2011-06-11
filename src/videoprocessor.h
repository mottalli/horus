#pragma once

#include "common.h"
#include "qualitychecker.h"
#include "segmentator.h"
#include "loggaborencoder.h"
#include "gaborencoder.h"

#include "eyedetect.h"

namespace horus {

class VideoProcessorParameters
{
public:
	int templateWaitTimeout;
	int focusThreshold;
	bool interlacedVideo;
	int correlationThreshold;
	float segmentationScoreThreshold;
	int minimumContourQuality;
	bool segmentEyelids;
	bool doEyeDetect;
	bool pauseAfterCapture;
	unsigned int pauseFrames;
	unsigned int minCountForTemplateAveraging;
	double minAverageTemplateQuality;

	VideoProcessorParameters()
	{
		this->templateWaitTimeout = 20;
		this->focusThreshold = 35;
		this->interlacedVideo = true;
		this->correlationThreshold = 92;
		this->segmentationScoreThreshold = 1.7;
		this->minimumContourQuality = 60;
		this->segmentEyelids = false;
		this->doEyeDetect = true;
		this->pauseAfterCapture = true;
		this->pauseFrames = 40;
		this->minCountForTemplateAveraging = 6;
		this->minAverageTemplateQuality = 70;
	}
};

typedef struct {
	GrayscaleImage image;
	SegmentationResult segmentationResult;
	IrisTemplate irisTemplate;
	double quality;
} CapturedTemplate;

class VideoProcessor {
public:
	VideoProcessor();
	virtual ~VideoProcessor();

	typedef enum {
		UNKNOWN_ERROR,
		UNPROCESSED,
		DEFOCUSED,
		INTERLACED,
		NO_EYE,
		FOCUSED_NO_IRIS,
		IRIS_LOW_QUALITY,
		IRIS_TOO_CLOSE,
		IRIS_TOO_FAR,
		FOCUSED_IRIS,
		BAD_TEMPLATE,
		FINISHED_CAPTURE,
		GOT_TEMPLATE
	} VideoStatus;

	enum {
		PROCTIME_INITIALIZE,
		PROCTIME_FOCUS_CHECK,
		PROCTIME_EYE_DETECT,
		PROCTIME_INTERLACE_CHECK,
		PROCTIME_SEGMENTATION,
		PROCTIME_IRIS_VALIDATION,
		PROCTIME_UNUSED				/* Just to know how many time slots we need */
	};

	std::vector<double> processingTime;

	VideoProcessorParameters parameters;

	VideoStatus processFrame(const Mat& frame);

	QualityChecker qualityChecker;
	Segmentator segmentator;
	LogGaborEncoder irisEncoder;
	//GaborEncoder irisEncoder;

	double lastFocusScore;
	QualityChecker::ValidationHeuristics lastIrisHeuristics;
	VideoStatus lastStatus;
	SegmentationResult lastSegmentationResult;
	double lastIrisQuality;
	IrisTemplate lastTemplate;

	IrisTemplate getAverageTemplate() const;
	GrayscaleImage getBestTemplateFrame() const;
	const SegmentationResult& getBestTemplateSegmentation() const;
	IrisTemplate getBestTemplate() const;

	Mat lastFrame;

	Rect eyeROI;

	vector<CapturedTemplate> templateBuffer;

private:
	GrayscaleImage lastFrameBW;

	EyeDetect eyeDetect;

	VideoStatus doProcess(const GrayscaleImage& image);

	Mat templateFrame;
	SegmentationResult templateSegmentation;
	double templateIrisQuality;

	int templateWaitCount;
	int framesToSkip;
	bool waitingTemplate;
	size_t bestTemplateIdx;

	inline void resetCapture()
	{
		this->templateWaitCount = 0;
		this->waitingTemplate = false;
		this->bestTemplateIdx = -1;
		this->templateBuffer.clear();
	}

	CascadeClassifier eyeClassifier;

};

}
