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
	int r = irisCircle.radius * 2.5;
	int x0 = std::max(0, irisCircle.xc-r);
	int x1 = std::min(image->width, irisCircle.xc+r);
	int y0 = std::max(0, irisCircle.yc-r);
	int y1 = std::min(image->height, irisCircle.yc+r);

	cvGetSubRect(image, &matWorkingRegion, cvRect(x0, y0, x1-x0, y1-y0));
	IplImage workingRegion;
	cvGetImage(&matWorkingRegion, &workingRegion);

	Parabola upperEyelid = this->segmentUpper(&workingRegion, pupilCircle.radius);
	Parabola lowerEyelid = this->segmentLower(&workingRegion, pupilCircle.radius);

	std::pair<Parabola, Parabola> result;
	result.first = Parabola(upperEyelid.x0+x0, upperEyelid.y0+y0, upperEyelid.p);
	result.second = Parabola(lowerEyelid.x0+x0, lowerEyelid.y0+y0, lowerEyelid.p);
	return result;
}

Parabola EyelidSegmentator::segmentUpper(const Image* image, int pupilRadius)
{
	CvMat tmp;
	cvGetSubRect(image, &tmp, cvRect(0, 0, image->width, image->height/2));
	//TODO: improve this so we don't have to cvCreateImage() every time
	Image* upperPart = cvCreateImage(cvGetSize(&tmp), IPL_DEPTH_32F, 1);
	cvConvert(&tmp, upperPart);

	Parabola bestParabola;
	double maxGrad = INT_MIN;

	for (int p = 150; p < 300; p += 50) {
	//for (int p = 200; p < 201; p++) {
		//std::pair<Parabola, double> res = this->findParabola(upperPart, p, 0, upperPart->height-pupilRadius);
		std::pair<Parabola, double> res = this->findParabola(upperPart, p, 0, upperPart->height);
		if (res.second > maxGrad) {
			maxGrad = res.second;
			bestParabola = res.first;
		}
	}

	cvReleaseImage(&upperPart);
	return bestParabola;
}

Parabola EyelidSegmentator::segmentLower(const Image* image, int pupilRadius)
{
	CvMat tmp;
	cvGetSubRect(image, &tmp, cvRect(0, image->height/2, image->width, image->height/2));
	//TODO: improve this so we don't have to cvCreateImage() every time
	Image* lowerPart = cvCreateImage(cvGetSize(&tmp), IPL_DEPTH_32F, 1);
	cvConvert(&tmp, lowerPart);

	Parabola bestParabola;
	double maxGrad = INT_MIN;

	for (int p = -300; p < -150; p += 50) {
		//std::pair<Parabola, double> res = this->findParabola(lowerPart, p, pupilRadius, lowerPart->height);
		std::pair<Parabola, double> res = this->findParabola(lowerPart, p, 0, lowerPart->height);
		if (res.second > maxGrad) {
			maxGrad = res.second;
			bestParabola = res.first;
		}
	}

	cvReleaseImage(&lowerPart);
	return Parabola(bestParabola.x0, bestParabola.y0+image->height/2, bestParabola.p);
}

std::pair<Parabola, double> EyelidSegmentator::findParabola(const Image* image, int p, int yMin, int yMax)
{
	IplImage* gradient = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
	int xMin = 0;
	int xMax = image->width-1;

	int step = Parameters::getParameters()->parabolicDetectorStep;

	CvMat* M = cvCreateMat((yMax-yMin)/step, (xMax-xMin)/step, CV_32F);

	cvSobel(image, gradient, 0, 1, 7);
	cvSmooth(gradient, gradient, CV_BLUR, 15);

	for (int i = 0; i < M->rows; i++) {
		int y0 = yMin+i*step;
		for (int j = 0; j < M->cols; j++) {
			int x0 = xMin+j*step;
			double avg = this->parabolaAverage(gradient, image, Parabola(x0, y0, p));
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

	int x0 = xMin+maxPos.x*step;
	int y0 = yMin+maxPos.y*step;

	return std::pair<Parabola, double>(Parabola(x0, y0, p), max);
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
