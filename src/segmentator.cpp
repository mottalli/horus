/*
 * File:   segmentator.cpp
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:37 PM
 */


#include <stdexcept>
#include <iostream>

#include "segmentator.h"
#include "helperfunctions.h"


Segmentator::Segmentator()
{
}

Segmentator::~Segmentator() {
}

SegmentationResult Segmentator::segmentImage(const Mat& image) {
	SegmentationResult result;
	Mat imageToSegment;		// Can either user image or workingImage depending on image format

	assert(image.depth() == CV_8U);

	if (image.channels() == 1) {
		imageToSegment = image;
	} else if (image.channels() == 3) {
		cvtColor(image, this->workingImage, CV_BGR2GRAY);
		imageToSegment = this->workingImage;
	}

	ContourAndCloseCircle pupilResult = this->pupilSegmentator.segmentPupil(imageToSegment);
	ContourAndCloseCircle irisResult = this->irisSegmentator.segmentIris(imageToSegment, pupilResult);

	result.pupilContour = pupilResult.first;
	result.pupilCircle = pupilResult.second;
	result.irisContour = irisResult.first;
	result.irisCircle = irisResult.second;
	result.pupilContourQuality = this->pupilSegmentator.getPupilContourQuality();

	result.eyelidsSegmented = false;

	return result;
};

void Segmentator::segmentEyelids(const Mat& image, SegmentationResult& result)
{
	Mat imageToSegment;		// Can either user image or workingImage depending on image format

	assert(image.depth() == IPL_DEPTH_8U);

	if (image.channels() == 1) {
		imageToSegment = image;
	} else if (image.channels() == 3) {
		cvtColor(image, this->workingImage, CV_BGR2GRAY);
		imageToSegment = this->workingImage;
	}

	std::pair<Parabola, Parabola> eyelids = this->eyelidSegmentator.segmentEyelids(&IplImage(imageToSegment), result.pupilCircle, result.irisCircle);
	result.upperEyelid = eyelids.first;
	result.lowerEyelid = eyelids.second;
	result.eyelidsSegmented = true;
}
