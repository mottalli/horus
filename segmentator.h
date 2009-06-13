/*
 * File:   segmentator.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:37 PM
 */

#ifndef _SEGMENTATOR_H
#define	_SEGMENTATOR_H

#include "common.h"
#include "pupilsegmentator.h"
#include "irissegmentator.h"
#include "eyelidsegmentator.h"
#include "segmentationresult.h"

class Segmentator {
public:
	Segmentator();
	virtual ~Segmentator();
	SegmentationResult segmentImage(const Image* image);
	void segmentEyelids(const Image* image, SegmentationResult& result);

	struct {
		Image* workingImage;
		float resizeFactor;
	} buffers;

	//private:
	PupilSegmentator _pupilSegmentator;
	IrisSegmentator _irisSegmentator;
	EyelidSegmentator _eyelidSegmentator;

	void setupBuffers(const Image* image);
};

#endif	/* _SEGMENTATOR_H */

