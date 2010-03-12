/*
 * File:   segmentator.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:37 PM
 */

#pragma once

#include "common.h"
#include "pupilsegmentator.h"
#include "irissegmentator.h"
#include "eyelidsegmentator.h"
#include "segmentationresult.h"

class Segmentator {
public:
	Segmentator();
	virtual ~Segmentator();
	SegmentationResult segmentImage(const IplImage* image);
	void segmentEyelids(const IplImage* image, SegmentationResult& result);

	struct {
		IplImage* workingImage;
		float resizeFactor;
	} buffers;

	//private:
	PupilSegmentator _pupilSegmentator;
	IrisSegmentator _irisSegmentator;
	EyelidSegmentator _eyelidSegmentator;

	void setupBuffers(const IplImage* image);
};


