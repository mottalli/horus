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
    cvSmooth(this->buffers.similarityImage, this->buffers.similarityImage, CV_GAUSSIAN, 13);

    // Now perform the cascaded integro-differential operator
    return this->cascadedIntegroDifferentialOperator(this->buffers.similarityImage);
    
}

Circle PupilSegmentator::cascadedIntegroDifferentialOperator(const Image* image) {
    int minrad = 10, minradabs = 10;
    int maxrad = 80;
    int minx = 10, miny = 10;
    int maxx = image->width-10, maxy = image->height-10;
    int i, x, y, radius;
    //int maxStep = INT_MIN;
    int bestX, bestY, bestRadius;

    std::vector<int> steps(3), radiusSteps(3);
    steps[0] = 10; steps[1] = 3; steps[2] = 1;
    radiusSteps[0] = 15; radiusSteps[1] = 3; radiusSteps[2] = 1;

    /*std::vector<int> steps(1), radiusSteps(1);
    radiusSteps[0] = steps[0] = 1;*/

    for (i = 0; i < steps.size(); i++) {
        int maxStep = INT_MIN;
        for (x = minx; x < maxx; x += steps[i]) {
            for (y = miny; y < maxy; y += steps[i]) {
                MaxAvgRadiusResult res = this->maxAvgRadius(image, x, y, minrad, maxrad, radiusSteps[i]);
                if (res.maxStep > maxStep) {
                    maxStep = res.maxStep;
                    bestX = x;
                    bestY = y;
                    bestRadius = res.maxRad;
                }
            }
        }

        //std::cout << i << " " << bestX << " " << bestY << " " << bestRadius << " " << maxStep << std::endl;

        minx = std::max<int>(bestX-steps[i], 0);
        maxx = std::min<int>(bestX+steps[i], image->width);
        miny = std::max<int>(bestY-steps[i], 0);
        maxy = std::min<int>(bestY+steps[i], image->height);
        minrad = std::max<int>(bestRadius-radiusSteps[i], minradabs);
        maxrad = bestRadius+radiusSteps[i];

    }

    Circle bestCircle;
    bestCircle.xc = bestX;
    bestCircle.yc = bestY;
    bestCircle.radius = bestRadius;

    return bestCircle;
}

PupilSegmentator::MaxAvgRadiusResult PupilSegmentator::maxAvgRadius(const Image* image, int x, int y, int radmin, int radmax, int radstep)
{
    int maxDifference, difference;
    unsigned char actualAvg, nextAvg;
    MaxAvgRadiusResult result;

    maxDifference = INT_MIN;

    actualAvg = this->circleAverage(image, x, y, radmin);
    for (int radius = radmin; radius <= radmax-radstep; radius += radstep) {
        nextAvg = this->circleAverage(image, x, y, radius+radstep);
        difference = int(actualAvg)-int(nextAvg);
        if (difference > maxDifference) {
            maxDifference = difference;
            result.maxRad = radius;
        }

        actualAvg = nextAvg;
    }

    result.maxStep = maxDifference;

    return result;
}

unsigned char PupilSegmentator::circleAverage(const Image* image, int xc, int yc, int rc) {
    // Optimized Bresenham algorithm for circles
    int x = 0;
    int y = rc;
    int d = 3 - 2 * rc;
    int i, w;
    unsigned char *row1, *row2, *row3, *row4;
    unsigned S, n;

    i = 0;
    n = 0;
    S = 0;

    if ((xc + rc) >= image->width || (xc - rc) < 0 || (yc + rc) >= image->height || (yc - rc) < 0) {
        while (x < y) {
            i++;
            w = (i - 1)*8 + 1;

            row1 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc + y);
            row2 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc - y);
            row3 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc + x);
            row4 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc - x);

            bool row1in = ((yc + y) >= 0 && (yc + y) < image->height);
            bool row2in = ((yc - y) >= 0 && (yc - y) < image->height);
            bool row3in = ((yc + x) >= 0 && (yc + x) < image->height);
            bool row4in = ((yc - x) >= 0 && (yc - y) < image->height);
            bool xcMxin = ((xc + x) >= 0 && (xc + x) < image->width);
            bool xcmxin = ((xc - x) >= 0 && (xc - x) < image->width);
            bool xcMyin = ((xc + y) >= 0 && (xc + y) < image->width);
            bool xcmyin = ((xc - y) >= 0 && (xc - y) < image->width);

            if (row1in && xcMxin) {
                S += unsigned(row1[xc + x]);
                n++;
            }
            if (row1in && xcmxin) {
                S += unsigned(row1[xc - x]);
                n++;

            }
            if (row2in && xcMxin) {
                S += unsigned(row2[xc + x]);
                n++;
            }
            if (row2in && xcmxin) {
                S += unsigned(row2[xc - x]);
                n++;
            }
            if (row3in && xcMyin) {
                S += unsigned(row3[xc + y]);
                n++;
            }
            if (row3in && xcmyin) {
                S += unsigned(row3[xc - y]);
                n++;
            }
            if (row4in && xcMyin) {
                S += unsigned(row4[xc + y]);
                n++;
            }
            if (row4in && xcmyin) {
                S += unsigned(row4[xc - y]);
                n++;
            }

            if (d < 0) {
                d = d + (4 * x) + 6;
            } else {
                d = d + 4 * (x - y) + 10;
                y--;
            }

            x++;
        }
    } else {
        while (x < y) {
            i++;
            w = (i - 1)*8 + 1;

            row1 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc + y);
            row2 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc - y);
            row3 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc + x);
            row4 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc - x);

            S += unsigned(row1[xc + x]);
            S += unsigned(row1[xc - x]);
            S += unsigned(row2[xc + x]);
            S += unsigned(row2[xc - x]);
            S += unsigned(row3[xc + y]);
            S += unsigned(row3[xc - y]);
            S += unsigned(row4[xc + y]);
            S += unsigned(row4[xc - y]);
            n += 8;

            if (d < 0) {
                d = d + (4 * x) + 6;
            } else {
                d = d + 4 * (x - y) + 10;
                y--;
            }

            x++;
        }
    }

    return (unsigned char) (S / n);
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
