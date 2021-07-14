#include <stdexcept>
#include <iostream>

#include "segmentator.h"
#include "tools.h"

using namespace horus;

Segmentator::Segmentator()
{
}

Segmentator::~Segmentator() {
}

SegmentationResult Segmentator::segmentImage(const Image& image, cv::Rect ROI)
{
    assert(image.depth() == CV_8U);

    timer.restart();

    SegmentationResult result;

    GrayscaleImage imageBW;
    tools::toGrayscale(image, imageBW, true);

    GrayscaleImage imageROI;
    bool hasROI = (ROI.width > 0);
    if (hasROI) {
        imageROI = imageBW(ROI);
    } else {
        imageROI = imageBW;
    }

    ContourAndCloseCircle pupilResult = this->pupilSegmentator.segmentPupil(imageROI);

    // If there was a ROI, adjust the coordinates of the result
    if (hasROI) {
        Contour& pupilContour = pupilResult.first;
        for (cv::Point& p : pupilContour) {
            p.x += ROI.x;
            p.y += ROI.y;
        }

        Circle& pupilCircle = pupilResult.second;
        pupilCircle.center.x += ROI.x;
        pupilCircle.center.y += ROI.y;
    }

    ContourAndCloseCircle irisResult = this->irisSegmentator.segmentIris(imageBW, pupilResult);

    result.pupilContour = pupilResult.first;
    result.pupilCircle = pupilResult.second;
    result.irisContour = irisResult.first;
    result.irisCircle = irisResult.second;
    result.pupilContourQuality = this->pupilSegmentator.getPupilContourQuality();

    result.eyelidsSegmented = false;

    this->segmentationTime = timer.elapsed();

    return result;
}

void Segmentator::segmentEyelids(const Image& image, SegmentationResult& result)
{
    GrayscaleImage imageBW;
    tools::toGrayscale(image, imageBW, false);

    assert(image.depth() == CV_8U);

    std::pair<Parabola, Parabola> eyelids = this->eyelidSegmentator.segmentEyelids(imageBW, result.pupilCircle, result.irisCircle);
    result.upperEyelid = eyelids.first;
    result.lowerEyelid = eyelids.second;
    result.eyelidsSegmented = true;
}
