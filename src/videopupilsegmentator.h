/*
 * File:   videopupils.h
 * Author: marcelo
 *
*/

#pragma once

#include "common.h"

namespace horus {

class VideoPupilSegmentatorParameters
{
public:
    int bufferWidth;
    double muPupil;
    double sigmaPupil;
    int minimumPupilRadius;
    int maximumPupilRadius;
    int pupilAdjustmentRingWidth;
    int pupilAdjustmentRingHeight;
    int infraredThreshold;
    bool avoidPupilReflection;

    VideoPupilSegmentatorParameters()
    {
        this->bufferWidth = 320;
        this->muPupil = 8.0;
        this->sigmaPupil = 6.0;
        this->minimumPupilRadius = 7;
        this->maximumPupilRadius = 80;
        this->pupilAdjustmentRingWidth = 256;
        this->pupilAdjustmentRingHeight = 80;
        this->infraredThreshold = 200;
        this->avoidPupilReflection = true;
    }
};

class VideoPupilSegmentator {
public:
    VideoPupilSegmentator();
    VideoPupilSegmentator(VideoPupilSegmentatorParameters params);
    virtual ~VideoPupilSegmentator();

    ContourAndCloseCircle segmentPupil(const GrayscaleImage& image);
    inline int getPupilContourQuality() const { return this->pupilContourQuality; }

    // Internal buffers
    GrayscaleImage similarityImage;
    GrayscaleImage equalizedImage;
    GrayscaleImage adjustmentRing;
    Mat_<int16_t> adjustmentRingGradient;
    GrayscaleImage workingImage;
    Mat1f adjustmentSnake;
    Mat1f originalAdjustmentSnake;
    GrayscaleImage _LUT;
    double resizeFactor;

    VideoPupilSegmentatorParameters parameters;

private:
    void setupBuffers(const Image& image);
    void similarityTransform(const GrayscaleImage& src, GrayscaleImage& dest);
    void postprocessSimilarity(GrayscaleImage& similarity);
    Circle approximatePupil(const GrayscaleImage& image);
    Circle cascadedIntegroDifferentialOperator(const GrayscaleImage& image);
    int calculatePupilContourQuality(const GrayscaleImage& region, const Mat_<uint16_t>& regionGradient, const Mat_<float>& contourSnake);

    int pupilContourQuality;

    typedef struct {
        int maxRad;
        int maxStep;
    } MaxAvgRadiusResult;
    MaxAvgRadiusResult maxAvgRadius(const GrayscaleImage& image, int x, int y, int radmin, int radmax, int radstep);

    uint8_t circleAverage(const GrayscaleImage& image, int x, int y, int radius);
    Contour adjustPupilContour(const GrayscaleImage& image, const Circle& approximateCircle);

    double _lastSigma, _lastMu;
    GrayscaleImage matStructElem;
};


};
