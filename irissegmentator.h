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
        Image* adjustmentRing;
        Image* adjustmentRingGradient;
        CvMat* adjustmentSnake;
    } buffers;

	ContourAndCloseCircle segmentIris(const Image* image, const ContourAndCloseCircle& pupilSegmentation);

private:
	void setupBuffers(const Image* image);
	ContourAndCloseCircle segmentIrisRecursive(const Image* image, const ContourAndCloseCircle& pupilSegmentation, int radiusMax=-1, int radiusMin=-1);

};


