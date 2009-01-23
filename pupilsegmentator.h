/* 
 * File:   pupilsegmentator.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:39 PM
 */

#ifndef _PUPILSEGMENTATOR_H
#define	_PUPILSEGMENTATOR_H

#include "common.h"


class PupilSegmentator {
public:
    PupilSegmentator();
    virtual ~PupilSegmentator();

    ContourAndCloseCircle segmentPupil(const Image* image);

    struct {
        Image* similarityImage;
        Image* equalizedImage;
        Image* adjustmentRing;
        Image* adjustmentRingGradient;
        CvMat* LUT;
    } buffers;

private:
    void setupBuffers(const Image* image);
    void similarityTransform();
    Circle approximatePupil(const Image* image);
    Circle cascadedIntegroDifferentialOperator();

    double _lastSigma, _lastMu;

};

#endif	/* _PUPILSEGMENTATOR_H */

