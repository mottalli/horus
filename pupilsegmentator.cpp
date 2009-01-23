/* 
 * File:   pupilsegmentator.cpp
 * Author: marcelo
 * 
 * Created on January 21, 2009, 8:39 PM
 */

#include "pupilsegmentator.h"

PupilSegmentator::PupilSegmentator() {
    this->buffers.LUT = cvCreateMat(256, 1, CV_8UC1);
    this->buffers.similarityImage = NULL;
    this->_lastSigma = this->_lastMu = -100.0;
}

PupilSegmentator::~PupilSegmentator() {
    cvReleaseMat(&this->buffers.LUT);

    if (this->buffers.similarityImage != NULL) {
        cvReleaseImage(&this->buffers.similarityImage);
    }
}

ContourAndCloseCircle PupilSegmentator::segmentPupil(const Image* image) {
    this->setupBuffers(image);
    ContourAndCloseCircle result;

    Circle pupilCircle = this->approximatePupil(image);

    result.second = pupilCircle;

    return result;

}

void PupilSegmentator::setupBuffers(const Image* image) {
    if (this->buffers.similarityImage != NULL) {
        CvSize currentBufferSize = cvGetSize(this->buffers.similarityImage);
        CvSize imageSize = cvGetSize(image);
        if (imageSize.height == currentBufferSize.height && imageSize.width == currentBufferSize.width) {
            // Must not update the buffers
            return;
        }

        cvReleaseImage(&this->buffers.similarityImage);
        cvReleaseImage(&this->buffers.equalizedImage);
    }

    this->buffers.similarityImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
    this->buffers.equalizedImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
}

Circle PupilSegmentator::approximatePupil(const Image* image) {
    // First, equalize the image
    cvEqualizeHist(image, this->buffers.equalizedImage);

    // Then apply the similarity transformation
    this->similarityTransform();

    // Now perform the cascaded integro-differential operator
    this->cascadedIntegroDifferentialOperator();
}

Circle PupilSegmentator::cascadedIntegroDifferentialOperator() {
    Image* image = this->buffers.similarityImage;
}

void PupilSegmentator::similarityTransform() {
    Parameters* parameters = Parameters::getParameters();

    double sigma = parameters->sigmaPupil;
    double mu = parameters->muPupil;
    double num, denom = 2.0*sigma*sigma;
    double res;

    if (this->_lastSigma != sigma || this->_lastMu != mu) {
        // Rebuild the lookup table
        this->_lastSigma = sigma;
        this->_lastMu = mu;

        unsigned char* pLUT = this->buffers.LUT->data.ptr;
        for (int i = 0; i < 256; i++) {
            num = (double(i)-mu)*(double(i)-mu);
            res = std::exp(-num/denom)*255.0;
            pLUT[i] = (unsigned char)(res);
        }
    }

    cvLUT(this->buffers.equalizedImage, this->buffers.similarityImage, this->buffers.LUT);
}
