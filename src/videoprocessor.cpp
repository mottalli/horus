/*
 * videoprocessor.cpp
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#include "videoprocessor.h"
#include "tools.h"

using namespace horus;
using namespace std;

VideoProcessor::VideoProcessor() : mtx(new boost::mutex())
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
    boost::mutex::scoped_lock lock(*this->mtx);

    this->processingTime = vector<double>(PROCTIME_UNUSED, 0.0);			// Initialize the timers to 0

    try {
        /***************** Initialization ***************/
        Timer t;
        frame.copyTo(this->lastFrame);

        if (this->framesToSkip > 0) {
            this->framesToSkip--;
            this->lastStatus = UNPROCESSED;
            return UNPROCESSED;
        }

        tools::toGrayscale(this->lastFrame, this->lastFrameBW, false);

        this->processingTime[PROCTIME_INITIALIZE] = t.elapsed();


        /***************** Quality checking ***************/

        this->lastStatus = this->doProcess(this->lastFrameBW);

        if (this->lastStatus == FOCUSED_IRIS && !this->waitingTemplate) {		// A new iris has been detected. Start the template generation process
            this->resetCapture();
            this->waitingTemplate = true;
        }

        if (this->waitingTemplate) {
            this->templateWaitCount++;

            if (this->lastStatus == FOCUSED_IRIS) {
                CapturedImage capturedImage;
                capturedImage.image = this->lastFrameBW.clone();

                // Delete this!
                LogGaborEncoder irisEncoder_;
                capturedImage.irisTemplate = irisEncoder_.generateTemplate(this->lastFrameBW, this->lastSegmentationResult);
                //capturedImage.irisTemplate = this->irisEncoder.generateTemplate(this->lastFrameBW, this->lastSegmentationResult);

                capturedImage.irisTemplate.irisQuality = this->lastSegmentationResult.pupilContourQuality;
                // Note that the "template quality" does not neccesarily have to be the valid bit count (could be something else)
                // That's why we put it outside the IrisTemplate class
                capturedImage.irisTemplate.templateQuality = capturedImage.irisTemplate.getValidBitCount();
                capturedImage.irisQuality = this->lastIrisQuality;
                capturedImage.segmentationResult = this->lastSegmentationResult;

                this->captureBurst.push_back(capturedImage);

                if (this->captureBurst.size() == 1) {
                    this->bestTemplateIdx = 0;
                } else if (this->lastIrisQuality > this->captureBurst[this->bestTemplateIdx].irisQuality) {		// Got a better quality image
                    this->bestTemplateIdx = captureBurst.size()-1;
                }

                this->lastTemplate = capturedImage.irisTemplate;

                if (this->captureBurst.size() < this->parameters.minCountForTemplateAveraging) {
                    this->templateWaitCount = 0;			// Reset the wait count (still capturing)
                } else {
                    this->lastStatus = FINISHED_CAPTURE;
                }
            }

            if (this->templateWaitCount >= this->parameters.templateWaitTimeout) {		// Finished the burst
                if (this->captureBurst.size() >= this->parameters.minCountForTemplateAveraging) {
                    // Enough templates - Can calculate average!
                    // Check if we have a "good" template
                    this->lastCapturedTemplate = this->getAverageTemplate();
                    if (this->qualityChecker.irisTemplateQuality(this->lastCapturedTemplate) < this->parameters.minAverageTemplateQuality) {
                        // Bad template - try another one (repeats the whole process!)
                        this->lastStatus = BAD_TEMPLATE;
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
    } catch (exception ex) {
        this->lastStatus = UNKNOWN_ERROR;
        cerr << ex.what() << endl;
    }

    return this->lastStatus;
}

VideoProcessor::VideoStatus VideoProcessor::doProcess(const GrayscaleImage& image)
{
    VideoStatus status = UNPROCESSED;


    /***************** 1. Check that the image is in focus ***************/
    Timer t1;
    this->lastFocusScore = this->qualityChecker.checkFocus(image);
    if (this->lastFocusScore < this->parameters.focusThreshold) {
        status = DEFOCUSED;
    }
    this->processingTime[PROCTIME_FOCUS_CHECK] = t1.elapsed();
    if (status != UNPROCESSED) return status;


    /***************** 2. Check that an eye is present in the image ***************/
    Timer t2;
    if (this->parameters.doEyeDetect) {
        // Step 2 - See if there's an eye
        if (!this->eyeDetect.detectEye(image)) {
            this->eyeROI = Rect();
            status = NO_EYE;
        } else {
            // Eye detected
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
        }
    } else {
        this->eyeROI = Rect();
    }
    this->processingTime[PROCTIME_EYE_DETECT] = t2.elapsed();
    if (status != UNPROCESSED) return status;


    /***************** 3. Check that the frame is not interlaced ***************/
    Timer t3;
    if (this->parameters.interlacedVideo) {
        double interlacedCorrelation = this->qualityChecker.interlacedCorrelation(image);
        if (interlacedCorrelation < this->parameters.correlationThreshold) {
            status = INTERLACED;
        }
    }
    this->processingTime[PROCTIME_INTERLACE_CHECK] = t3.elapsed();
    if (status != UNPROCESSED) return status;


    /***************** 4. Segment the frame  ***************/
    Timer t4;
    this->lastSegmentationResult = segmentator.segmentImage(image, this->eyeROI);
    if (this->parameters.segmentEyelids) {
        segmentator.segmentEyelids(image, this->lastSegmentationResult);
    }
    this->processingTime[PROCTIME_SEGMENTATION] = t4.elapsed();


    /***************** 5. Validate iris   ***************/
    Timer t5;
    this->lastIrisHeuristics = qualityChecker.validateIris(image, this->lastSegmentationResult);
    if (this->lastIrisHeuristics != QualityChecker::HAS_IRIS) {
        // No iris found on the image, or the segmentation is incorrect
        status = FOCUSED_NO_IRIS;
    } else {
        this->lastIrisQuality = qualityChecker.getIrisQuality(image, this->lastSegmentationResult);
        if (this->lastIrisQuality < this->parameters.minimumContourQuality) {
            // The image is kind of focused but the iris doesn't have enough quality
            // TODO: Check whether to return IRIS_TOO_FAR or IRIS_TOO_CLOSE

            status = IRIS_LOW_QUALITY;
        }
    }
    this->processingTime[PROCTIME_IRIS_VALIDATION] = t5.elapsed();
    if (status != UNPROCESSED) return status;

    // At this point we have a good quality image and we have enough reasons to believe
    // it's properly segmented.
    return FOCUSED_IRIS;
}

IrisTemplate VideoProcessor::getBestTemplate() const
{
    return this->captureBurst[this->bestTemplateIdx].irisTemplate;
}

IrisTemplate VideoProcessor::getAverageTemplate() const
{
    vector<CapturedImage> buffer = this->captureBurst;			// Work with a copy
    vector<IrisTemplate> templates(buffer.size());
    for (size_t i = 0; i < buffer.size(); i++) {
        templates[i] = buffer[i].irisTemplate;
    }
    return IrisEncoder::averageTemplates(templates);
}

GrayscaleImage VideoProcessor::getBestTemplateFrame() const
{
    GrayscaleImage res;
    tools::stretchHistogram(this->captureBurst[this->bestTemplateIdx].image, res, 0.01, 0);
    return res;
    //return this->templateBuffer[this->bestTemplateIdx].image;
}

const SegmentationResult& VideoProcessor::getBestTemplateSegmentation() const
{
    boost::mutex::scoped_lock lock(*this->mtx);
    return this->captureBurst[this->bestTemplateIdx].segmentationResult;
}

IrisTemplate VideoProcessor::getCapturedTemplate() const
{
    boost::mutex::scoped_lock lock(*this->mtx);
    return this->lastCapturedTemplate;
}
