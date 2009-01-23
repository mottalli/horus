/* 
 * File:   segmentator.cpp
 * Author: marcelo
 * 
 * Created on January 21, 2009, 8:37 PM
 */


#include <stdexcept>

#include "segmentator.h"


Segmentator::Segmentator() {
    this->buffers.workingImage = NULL;
}

Segmentator::~Segmentator() {
}

SegmentationResult Segmentator::segmentImage(const Image* image) {
    SegmentationResult result;

    if (image->nChannels != 1 || image->depth != IPL_DEPTH_8U) {
        throw std::runtime_error("Segmentator::segmentImage: the image must be one channel, grayscale");
    }

    this->setupBuffers(image);

    ContourAndCloseCircle pupilResult = this->_pupilSegmentator.segmentPupil(this->buffers.workingImage);
    result.pupilContour = pupilResult.first;
    result.irisContour  = this->_irisSegmentator.segmentIris(this->buffers.workingImage, pupilResult);

    return result;
};

void Segmentator::setupBuffers(const Image* image) {
    int bufferWidth = Parameters::getParameters()->bufferWidth;

    int workingWidth, workingHeight;
    int width = image->width, height = image->height;
    float resizeFactor;

    if (image->width > bufferWidth) {
        resizeFactor = double(bufferWidth) / double(image->width);
        workingWidth = int(double(width)*resizeFactor);
        workingHeight = int(double(height)*resizeFactor);
    } else {
        resizeFactor = 1.0;
        workingWidth = width;
        workingHeight = height;
    }

    this->buffers.resizeFactor = resizeFactor;
    
    Image*& workingImage = this->buffers.workingImage;

    if (workingImage == NULL || workingImage->width != workingWidth || workingImage->height != workingHeight) {
        if (workingImage != NULL) {
            cvReleaseImage(&workingImage);
        }

        workingImage = cvCreateImage(cvSize(workingWidth, workingHeight), IPL_DEPTH_8U, 1);

        if (resizeFactor == 1.0) {
            cvCopy(image, workingImage);
        } else {
            cvResize(image, workingImage, CV_INTER_LINEAR);
        }
    }
}
