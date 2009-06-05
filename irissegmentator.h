/*
 * File:   irissegmentator.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:39 PM
 */

#ifndef _IRISSEGMENTATOR_H
#define	_IRISSEGMENTATOR_H

#include "types.h"

class IrisSegmentator {
public:
    IrisSegmentator();
    virtual ~IrisSegmentator();

    struct {
        Image* adjustmentRing;
        Image* adjustmentRingGradient;
        CvMat* adjustmentSnake;
    } buffers;

    ContourAndCloseCircle segmentIris(const Image* image, const ContourAndCloseCircle& pupilSegmentation);

private:
	void setupBuffers(const Image* image);

};

#endif	/* _IRISSEGMENTATOR_H */

