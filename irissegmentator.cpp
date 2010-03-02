/*
 * File:   irissegmentator.cpp
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:39 PM
 */

#include "irissegmentator.h"
#include "parameters.h"
#include "helperfunctions.h"
#include <cmath>

IrisSegmentator::IrisSegmentator() {
	this->buffers.adjustmentRing = NULL;
	this->buffers.adjustmentRingGradient = NULL;
}


IrisSegmentator::~IrisSegmentator() {
}

ContourAndCloseCircle IrisSegmentator::segmentIris(const IplImage* image, const ContourAndCloseCircle& pupilSegmentation)
{
	return this->segmentIrisRecursive(image, pupilSegmentation, -1, -1);
}

ContourAndCloseCircle IrisSegmentator::segmentIrisRecursive(const IplImage* image, const ContourAndCloseCircle& pupilSegmentation, int radiusMax, int radiusMin)
{
	this->setupBuffers(image);

	Circle pupilCircle = pupilSegmentation.second;
	if (radiusMin < 0) {
	    radiusMin = pupilCircle.radius * 1.3;
	}

	if (radiusMax < 0) {
	    radiusMax = pupilCircle.radius * 5.0;
	}

	HelperFunctions::extractRing(image, this->buffers.adjustmentRing, pupilCircle.xc, pupilCircle.yc, radiusMin, radiusMax);
	cvSmooth(this->buffers.adjustmentRing, this->buffers.adjustmentRing, CV_GAUSSIAN, 3, 15);
	cvSobel(this->buffers.adjustmentRing, this->buffers.adjustmentRingGradient, 0, 1, 3);
	//cvSmooth(this->buffers.adjustmentRingGradient, this->buffers.adjustmentRingGradient, CV_GAUSSIAN, 3, 15);

	double theta0 = -M_PI/4.0;
	double theta1 = M_PI/4.0;
	double theta2 = 3.0*M_PI/4.0;
	double theta3 = 5.0*M_PI/4.0;

	IplImage* gradient = this->buffers.adjustmentRingGradient;
	CvMat* snake = this->buffers.adjustmentSnake;

	assert(snake->width == this->buffers.adjustmentRing->width);

	int x0 = int((theta0/(2.0*M_PI))*double(snake->cols));
	int x1 = int((theta1/(2.0*M_PI))*double(snake->cols));
	int x2 = int((theta2/(2.0*M_PI))*double(snake->cols));
	int x3 = int((theta3/(2.0*M_PI))*double(snake->cols));

	assert((x1 * x2 * x3) > 0);		// x1, x2 and x3 must be positive (x0 may be negative)
	#define XIMAGE(x) ((x) >= 0 ? (x) : snake->cols+(x))

	int sumY1, maxSumY1 = INT_MIN, sumY2, maxSumY2 = INT_MIN;
	int bestY1 = 0, bestY2 = 0;

	assert(gradient->depth == IPL_DEPTH_16S);
	for (int y = 0; y < gradient->height; y++) {
		sumY1 = 0;
		sumY2 = 0;

		int16_t* row = (int16_t*)(gradient->imageData + y*gradient->widthStep);

		for (int x = x0; x < x1; x++) {
			sumY1 += row[XIMAGE(x)];
		}
		for (int x = x2; x < x3; x++) {
			sumY2 += row[x];		// No need to use XIMAGE here
		}

		if (sumY1 > maxSumY1) {
			maxSumY1 = sumY1;
			bestY1 = y;
		}
		if (sumY2 > maxSumY2) {
			maxSumY2 = sumY2;
			bestY2 = y;
		}
	}

	// If there's a big difference between bestY1 and bestY2, it probably means something wasn't right.
	if (abs(bestY1-bestY2) > gradient->height*0.2) {
		// Try again with a smaller radiusMax (less chances of error)
		radiusMax = radiusMax - pupilCircle.radius/2;
		if (radiusMax > radiusMin) {
			return this->segmentIrisRecursive(image, pupilSegmentation, radiusMax);
		}
	}


	for (int x = x0; x < x1; x++) {
		cvSetReal2D(snake, 0, XIMAGE(x), bestY1);
	}

	for (int x = x2; x < x3; x++) {
		cvSetReal2D(snake, 0, x, bestY2);
	}

	// Interpolation between the two segments
	for (int x = x1; x < x2; x++) {
		int y = bestY1 + (double(x-x1)/double(x2-1-x1)) * double(bestY2-bestY1);
		cvSetReal2D(snake, 0, x, y);
	}
	for (int x = x3; x < XIMAGE(x0) && x < snake->width; x++) {
		int y = bestY2 + (double(x-x3)/double(snake->width-1-x3)) * double(bestY1-bestY2);
		cvSetReal2D(snake, 0, x, y);
	}

	for (int x = 0; x < snake->width; x++) {
		int maxGrad = INT_MIN;
		int maxY = 0;
		int v = cvGetReal2D(snake, 0, x);
		int y0 = std::max(0, v-5);
		int y1 = std::min(gradient->height, v+5);

		for (int y = y0; y < y1; y++) {
			int gxy = cvGetReal2D(gradient,y,x);
			if (gxy > maxGrad) {
				maxGrad = gxy;
				maxY = y;
			}
		}
		cvSetReal2D(snake, 0, x, maxY);
	}

	// Smooth the snake
	HelperFunctions::smoothSnakeFourier(snake, 3);

	// Convert to image coordinates
	Contour irisContour(snake->width);
	for (int x = 0; x < snake->width; x++) {
		double theta = (double(x)/double(snake->width))*2.0*M_PI;
		double radius = ((double(cvGetReal2D(snake, 0, x))/double(gradient->height-1))*double(radiusMax-radiusMin)) + double(radiusMin);
		int ximage = int(double(pupilCircle.xc) + std::cos(theta) * radius);
		int yimage = int(double(pupilCircle.yc) + std::sin(theta) * radius);
		irisContour[x] = cvPoint(ximage, yimage);
	}

	ContourAndCloseCircle result;

	result.first = irisContour;
	result.second = HelperFunctions::approximateCircle(result.first);

    return result;
}

void IrisSegmentator::setupBuffers(const IplImage* image)
{
	Parameters* parameters = Parameters::getParameters();

    if (this->buffers.adjustmentRing == NULL || this->buffers.adjustmentRing->width != parameters->irisAdjustmentRingWidth || this->buffers.adjustmentRing->height != parameters->irisAdjustmentRingHeight) {
    	if (this->buffers.adjustmentRing != NULL) {
    		cvReleaseImage(&this->buffers.adjustmentRing);
    		cvReleaseImage(&this->buffers.adjustmentRingGradient);
			cvReleaseMat(&this->buffers.adjustmentSnake);
    	}

    	this->buffers.adjustmentRing = cvCreateImage(cvSize(parameters->irisAdjustmentRingWidth, parameters->irisAdjustmentRingHeight), IPL_DEPTH_16S, 1);
    	this->buffers.adjustmentRingGradient = cvCreateImage(cvSize(parameters->irisAdjustmentRingWidth, parameters->irisAdjustmentRingHeight), IPL_DEPTH_16S, 1);
    	this->buffers.adjustmentSnake = cvCreateMat(1, parameters->irisAdjustmentRingWidth, CV_32F);
    }
}
