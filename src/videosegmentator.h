/*
 * File:   segmentator.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:37 PM
 */

#pragma once

#include "common.h"
#include "videopupilsegmentator.h"
#include "irissegmentator.h"
#include "eyelidsegmentator.h"
#include "segmentator.h"
#include "clock.h"

namespace horus {

class VideoSegmentator {
public:
    VideoSegmentator();
    virtual ~VideoSegmentator();

    SegmentationResult segmentImage(const Mat& image, cv::Rect ROI=cv::Rect());

    VideoPupilSegmentator pupilSegmentator;
    IrisSegmentator irisSegmentator;

    double segmentationTime;

private:
    float resizeFactor;
    Timer timer;

    GrayscaleImage blurredImage;			// Used as a buffer to calculate the ROI
};

}
