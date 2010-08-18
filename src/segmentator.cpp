#include <stdexcept>
#include <iostream>

#include "segmentator.h"


Segmentator::Segmentator()
{
}

Segmentator::~Segmentator() {
}

SegmentationResult Segmentator::segmentImage(const Mat& image) {
	cout << "EN segmentImage" << endl;
	cout << image.size().width << endl;

	clock.start();
	
	cout << "1" << endl;

	SegmentationResult result;
	Mat imageToSegment;		// Can either user image or workingImage depending on image format
	
	cout << "2" << endl;

	assert(image.depth() == CV_8U);
	
	cout << "3" << endl;

	if (image.channels() == 1) {
		cout << "3.1.1" << endl;
		imageToSegment = image;
		cout << "3.1.2" << endl;
	} else if (image.channels() == 3) {
		cout << "3.2.1" << endl;
		cout << (void*)image.data << endl;
		cvtColor(image, this->workingImage, CV_BGR2GRAY);
		cout << "3.2.2" << endl;
		imageToSegment = this->workingImage;
		cout << "3.2.3" << endl;
	}
	
	cout << "4" << endl;

	ContourAndCloseCircle pupilResult = this->pupilSegmentator.segmentPupil(imageToSegment);
	ContourAndCloseCircle irisResult = this->irisSegmentator.segmentIris(imageToSegment, pupilResult);

	result.pupilContour = pupilResult.first;
	result.pupilCircle = pupilResult.second;
	result.irisContour = irisResult.first;
	result.irisCircle = irisResult.second;
	result.pupilContourQuality = this->pupilSegmentator.getPupilContourQuality();

	result.eyelidsSegmented = false;

	this->segmentationTime = clock.stop();

	return result;
};

void Segmentator::segmentEyelids(const Mat& image, SegmentationResult& result)
{
	Mat imageToSegment;		// Can either user image or workingImage depending on image format

	assert(image.depth() == CV_8U);

	if (image.channels() == 1) {
		imageToSegment = image;
	} else if (image.channels() == 3) {
		cvtColor(image, this->workingImage, CV_BGR2GRAY);
		imageToSegment = this->workingImage;
	}

	std::pair<Parabola, Parabola> eyelids = this->eyelidSegmentator.segmentEyelids(imageToSegment, result.pupilCircle, result.irisCircle);
	result.upperEyelid = eyelids.first;
	result.lowerEyelid = eyelids.second;
	result.eyelidsSegmented = true;
}
