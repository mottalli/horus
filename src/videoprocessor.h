#pragma once

#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

#include "common.h"
#include "qualitychecker.h"
#include "segmentator.h"
#include "loggaborencoder.h"
#include "gaborencoder.h"
#include "eyedetect.h"
#include "videosegmentator.h"

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
        this->templateWaitTimeout = 10;
        this->focusThreshold = 35;
        this->interlacedVideo = true;
        this->correlationThreshold = 92;
        this->segmentationScoreThreshold = 1.7f;
        this->minimumContourQuality = 60;
        this->segmentEyelids = false;
        this->doEyeDetect = false;
        this->pauseAfterCapture = true;
        this->pauseFrames = 40;
        this->minCountForTemplateAveraging = 10;
        this->minAverageTemplateQuality = 60;
    }
};

typedef struct {
    GrayscaleImage image;
    SegmentationResult segmentationResult;
    IrisTemplate irisTemplate;
    double irisQuality;
} CapturedImage;

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
        PROCTIME_UNUSED				/* Just to know how many proctime slots we need */
    };

    std::vector<double> processingTime;

    VideoProcessorParameters parameters;

    VideoStatus processFrame(const Mat& frame);

    QualityChecker qualityChecker;
    VideoSegmentator segmentator;
    LogGaborEncoder irisEncoder;
    //GaborEncoder irisEncoder;

    double lastFocusScore;
    QualityChecker::ValidationHeuristics lastIrisHeuristics;
    VideoStatus lastStatus;
    SegmentationResult lastSegmentationResult;
    double lastIrisQuality;
    IrisTemplate lastTemplate;

    GrayscaleImage getBestTemplateFrame() const;
    const SegmentationResult& getBestTemplateSegmentation() const;
    IrisTemplate getCapturedTemplate() const;
    Mat lastFrame;

    Rect eyeROI;

    typedef std::vector<CapturedImage> CaptureBurst;
    CaptureBurst captureBurst;

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
        this->captureBurst.clear();
    }

    CascadeClassifier eyeClassifier;

    IrisTemplate getAverageTemplate() const;
    IrisTemplate getBestTemplate() const;
    IrisTemplate lastCapturedTemplate;

    mutable boost::shared_ptr<boost::mutex> mtx;

};

}
