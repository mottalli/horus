/*
 * File:   irissegmentator.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:39 PM
 */

#pragma once

#include "types.h"

class IrisSegmentator {
public:
    IrisSegmentator();
    virtual ~IrisSegmentator();

    struct {
		IplImage* adjustmentRing;
		IplImage* adjustmentRingGradient;
        CvMat* adjustmentSnake;
    } buffers;

	ContourAndCloseCircle segmentIris(const IplImage* image, const ContourAndCloseCircle& pupilSegmentation);

private:
	void setupBuffers(const IplImage* image);
	ContourAndCloseCircle segmentIrisRecursive(const IplImage* image, const ContourAndCloseCircle& pupilSegmentation, int radiusMax=-1, int radiusMin=-1);

};


