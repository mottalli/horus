/*
 * eyelidsegmentator.cpp
 *
 *  Created on: Jun 5, 2009
 *      Author: marcelo
 */

#include "eyelidsegmentator.h"
#include "parameters.h"

EyelidSegmentator::EyelidSegmentator()
{

}

EyelidSegmentator::~EyelidSegmentator()
{
}

std::pair<Parabola, Parabola> EyelidSegmentator::segmentEyelids(const Image* image, const Circle& pupilCircle, const Circle& irisCircle)
{
	assert(image->origin == 0);

	CvMat matWorkingRegion;
	int r = irisCircle.radius * 1.5;
	int x0 = std::max(0, irisCircle.xc-r);
	int x1 = std::min(image->width, irisCircle.xc+r);
	int y0upper = std::max(0, irisCircle.yc-r);
	int y1upper = irisCircle.yc;
	int y0lower= irisCircle.yc;
	int y1lower = std::min(image->height, irisCircle.yc+r);

	Parabola upperEyelid = this->segmentUpper(image, x0, y0upper, x1, y1upper, pupilCircle, irisCircle);
	Parabola lowerEyelid = this->segmentLower(image, x0, y0lower, x1, y1lower, pupilCircle, irisCircle);

	std::pair<Parabola, Parabola> result;
	result.first = upperEyelid;
	result.second = lowerEyelid;
	return result;
}

Parabola EyelidSegmentator::segmentUpper(const Image* image, int x0, int y0, int x1, int y1, const Circle& pupilCircle, const Circle& irisCircle)
{
	Parabola bestParabola;
	double maxGrad = INT_MIN;

	for (int p = 150; p < 300; p += 50) {
		std::pair<Parabola, double> res = this->findParabola(image, p, x0, y0, x1, y1);
		if (res.second > maxGrad) {
			maxGrad = res.second;
			bestParabola = res.first;
		}
	}

	return bestParabola;
}

Parabola EyelidSegmentator::segmentLower(const Image* image, int x0, int y0, int x1, int y1, const Circle& pupilCircle, const Circle& irisCircle)
{
	Parabola bestParabola;
	double maxGrad = INT_MIN;

	for (int p = -300; p < -150; p += 50) {
		std::pair<Parabola, double> res = this->findParabola(image, p, x0, y0, x1, y1);
		if (res.second > maxGrad) {
			maxGrad = res.second;
			bestParabola = res.first;
		}
	}

	return bestParabola;
}

std::pair<Parabola, double> EyelidSegmentator::findParabola(const Image* image, int p, int x0, int y0, int x1, int y1)
{
	IplImage* gradient = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);

	int step = Parameters::getParameters()->parabolicDetectorStep;

	CvMat* M = cvCreateMat((y1-y0)/step, (x1-x0)/step, CV_32F);

	cvSobel(image, gradient, 0, 1, 7);
	cvSmooth(gradient, gradient, CV_BLUR, 15);

	for (int i = 0; i < M->rows; i++) {
		int y = y0+i*step;
		for (int j = 0; j < M->cols; j++) {
			int x = x0+j*step;
			double avg = this->parabolaAverage(gradient, image, Parabola(x, y, p));
			cvSetReal2D(M, i, j, avg);
		}
	}

	CvMat* Dx = cvCreateMat(M->rows, M->cols, CV_32F);
	CvMat* Dy = cvCreateMat(M->rows, M->cols, CV_32F);
	CvMat* D = cvCreateMat(M->rows, M->cols, CV_32F);

	cvSobel(M, Dx, 1, 0);
	cvSobel(M, Dy, 0, 1);

	// D = Dx^2 + Dy^2
	cvMul(Dx, Dx, Dx);
	cvMul(Dy, Dy, Dy);
	cvAdd(Dx, Dy, D);

	CvPoint maxPos;
	CvPoint minPos;
	double max;
	cvMinMaxLoc(D, &max, NULL, &minPos, &maxPos);

	cvReleaseImage(&gradient);
	cvReleaseMat(&M);
	cvReleaseMat(&Dx);
	cvReleaseMat(&Dy);
	cvReleaseMat(&D);

	int x = x0+maxPos.x*step;
	int y = y0+maxPos.y*step;

	return std::pair<Parabola, double>(Parabola(x, y, p), max);
}

double EyelidSegmentator::parabolaAverage(const Image* gradient, const Image* originalImage, const Parabola& parabola)
{
	double S = 0;
	int n = 0;
	double x, y;

	assert(originalImage->depth == IPL_DEPTH_32F);
	assert(gradient->depth == IPL_DEPTH_32F);

	for (x = 0; x < gradient->width; x += gradient->width/50 + 1) {
		y = parabola.value(x);
		if (y < 0 || y >= gradient->height) {
			continue;
		}

		float* rowGradient = (float*)(gradient->imageData + int(y)*gradient->widthStep);
		float* rowImage = (float*)(originalImage->imageData + int(y)*originalImage->widthStep);

		float v = rowImage[int(x)];
		if (v < 80 || v > 250) {
			// Try to avoid the pupil and the infrared reflection
			continue;
		}

		S += rowGradient[int(x)];
		n++;
	}

	if (!n) {
		return 0;		// No values were processed
	} else {
		return S/double(n);
	}
}
