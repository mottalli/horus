#include <stdexcept>
#include <iostream>

#include "segmentator.h"


Segmentator::Segmentator()
{
}

Segmentator::~Segmentator() {
}

SegmentationResult Segmentator::segmentImage(const Mat& image) {
	clock.start();
	
	SegmentationResult result;
	Mat imageToSegment;		// Can either user image or workingImage depending on image format
	
	assert(image.depth() == CV_8U);
	
	if (image.channels() == 1) {
		imageToSegment = image;
	} else if (image.channels() == 3) {
		cvtColor(image, this->workingImage, CV_BGR2GRAY);
		imageToSegment = this->workingImage;
	}

	this->ROI = this->calculateROI(imageToSegment);		// The Region of Interest excludes the black borders caused by hair or an out-of-frame head
	this->pupilSegmentator.setROI(this->ROI);

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

Rect Segmentator::calculateROI(const Mat_<uint8_t>& image)
{
	int mean, x0, x1, y0, y1, x, y;
	const int THRESH = 60;

	GaussianBlur(image, this->blurredImage, Size(7,7), 0);

	int rows = this->blurredImage.rows, cols = this->blurredImage.cols;


	for (x0 = 0, mean = 0; mean < THRESH && x0 < cols; x0++) {
		for (y = 0; y < rows; y++) {
			mean += int(this->blurredImage(y, x0));
		}
		mean = mean/cols;
	}

	for (x1 = cols-1, mean=0; mean < THRESH && x1 > x0+1; x1--) {
		for (y = 0; y < rows; y++) {
			mean += int(this->blurredImage(y, x1));
		}
		mean = mean/rows;
	}

	for (y0 = 0, mean = 0; mean < THRESH && y0 < rows; y0++) {
		for (x = 0; x < cols; x++) {
			mean += int(this->blurredImage(y0, x));
		}
		mean = mean/cols;
	}

	for (y1 = rows, mean=0; mean < THRESH && y1 > y0+1; y1--) {
		for (x = 0; x < cols; x++) {
			mean += int(this->blurredImage(y1, x));
		}
		mean = mean/cols;
	}

	return Rect(x0, y0, x1-x0, y1-y0);
}
