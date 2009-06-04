/*
 * File:   segmentator.cpp
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:37 PM
 */


#include <stdexcept>

#include "segmentator.h"


Segmentator::Segmentator()
{
}

Segmentator::~Segmentator() {
}

SegmentationResult Segmentator::segmentImage(const Image* image) {
    SegmentationResult result;

    if (image->nChannels != 1 || image->depth != IPL_DEPTH_8U) {
        throw std::runtime_error("Segmentator::segmentImage: the image must be one channel, grayscale");
    }

    ContourAndCloseCircle pupilResult = this->_pupilSegmentator.segmentPupil(image);
    result.pupilContour = pupilResult.first;
    result.pupilCircle = pupilResult.second;
    result.irisContour  = this->_irisSegmentator.segmentIris(image, pupilResult);

    return result;
};

void Segmentator::setupBuffers(const Image* image)
{

}
