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
#include "clock.h"

class Segmentator {
public:
	Segmentator();
	virtual ~Segmentator();

	SegmentationResult segmentImage(const Mat& image);
	void segmentEyelids(const Mat& image, SegmentationResult& result);

	PupilSegmentator pupilSegmentator;
	IrisSegmentator irisSegmentator;
	EyelidSegmentator eyelidSegmentator;

	double segmentationTime;

private:
	Mat workingImage;
	float resizeFactor;
	Clock clock;
};


