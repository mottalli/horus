#include <stdexcept>
#include <iostream>

#include "segmentator.h"
#include "tools.h"

Segmentator::Segmentator() :
	eyeROI(0,0,0,0)
{
}

Segmentator::~Segmentator() {
}

SegmentationResult Segmentator::segmentImage(const Image& image) {
	assert(image.depth() == CV_8U);

	clock.start();
	
	SegmentationResult result;
	
	GrayscaleImage imageBW;
	Tools::toGrayscale(image, imageBW, false);

	if (this->eyeROI.width > 0) {
		this->pupilSegmentator.setROI(this->eyeROI);
	}

	ContourAndCloseCircle pupilResult = this->pupilSegmentator.segmentPupil(imageBW);
	ContourAndCloseCircle irisResult = this->irisSegmentator.segmentIris(imageBW, pupilResult);

	result.pupilContour = pupilResult.first;
	result.pupilCircle = pupilResult.second;
	result.irisContour = irisResult.first;
	result.irisCircle = irisResult.second;
	result.pupilContourQuality = this->pupilSegmentator.getPupilContourQuality();

	result.eyelidsSegmented = false;

	this->segmentationTime = clock.stop();

	return result;
};

void Segmentator::segmentEyelids(const Image& image, SegmentationResult& result)
{
	GrayscaleImage imageBW;
	Tools::toGrayscale(image, imageBW, false);

	assert(image.depth() == CV_8U);

	std::pair<Parabola, Parabola> eyelids = this->eyelidSegmentator.segmentEyelids(imageBW, result.pupilCircle, result.irisCircle);
	result.upperEyelid = eyelids.first;
	result.lowerEyelid = eyelids.second;
	result.eyelidsSegmented = true;
}
