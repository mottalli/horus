/*
 * qualitychecker.cpp
 *
 *  Created on: Jun 19, 2009
 *      Author: marcelo
 */

#include "qualitychecker.h"
#include <cmath>

QualityChecker::QualityChecker()
{

}

QualityChecker::~QualityChecker(){
}

double QualityChecker::interlacedCorrelation(const Image* frame)
{
	CvSize size = cvGetSize(frame);
	Image* even = cvCreateImage(cvSize(size.width, size.height/2), IPL_DEPTH_8U, 1);
	Image* odd = cvCreateImage(cvSize(size.width, size.height/2), IPL_DEPTH_8U, 1);
	CvMat tmpodd, tmpeven, tmpdest;

	for (int i = 0; i < size.height/2; i++) {
		cvGetSubRect(frame, &tmpeven, cvRect(0, 2*i, size.width, 1));
		cvGetSubRect(frame, &tmpodd, cvRect(0, 2*i+1, size.width, 1));

		cvGetSubRect(even, &tmpdest, cvRect(0, i, size.width, 1));
		cvCopy(&tmpeven, &tmpdest);
		cvGetSubRect(odd, &tmpdest, cvRect(0, i, size.width, 1));
		cvCopy(&tmpodd, &tmpdest);
	}

	// TODO: Use buffers
	Image* bufX = cvCreateImage(cvGetSize(even), IPL_DEPTH_32F, 1);
	Image* bufY = cvCreateImage(cvGetSize(odd), IPL_DEPTH_32F, 1);
	Image* bufMul = cvCreateImage(cvGetSize(even), IPL_DEPTH_32F, 1);

	double meanX = 0, stdX = 0, meanY = 0, stdY = 0;
	double mean;

	cvConvert(even, bufX);
	cvConvert(odd, bufY);

	cvMean_StdDev(bufX, &meanX, &stdX);
	cvMean_StdDev(bufY, &meanY, &stdY);

	cvSubS(bufX, cvScalar(meanX,0,0,0), bufX);
	cvSubS(bufY, cvScalar(meanY,0,0,0), bufY);
	cvMul(bufX, bufY, bufMul);

	mean = cvMean(bufMul);

	cvReleaseImage(&bufX);
	cvReleaseImage(&bufY);
	cvReleaseImage(&bufMul);
	cvReleaseImage(&odd);
	cvReleaseImage(&even);

	return 100.0*mean/(stdX*stdY);

}

double QualityChecker::checkFocus(const Image* image)
{
	Image* src = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
	cvConvert(image, src);

	cvDCT(src, src, CV_DXT_FORWARD);
	cvAbs(src, src);

	int r0 = image->width/30;
	int r1 = image->width/15;

	double S0 = 0;
	double S1 = 0;

	int r02 = r0*r0;
	int r12 = r1*r1;
	for (int y = 1; y < r1; y++) {
		for (int x = 1; x < r1; x++) {
			if (x*x+y*y < r02) {
				S0 += cvGetReal2D(src, y, x);
			} else if (x*x+y*y < r12) {
				S1 += cvGetReal2D(src, y, x);
			}

		}
	}

	double q = S1/S0;
	double c = 0.6;

	cvReleaseImage(&src);
	return 100.0*q*q/(q*q+c*c);
}

/**
 * Checks if the segmentation is correct (heuristics - not 100% reliable)
 */
bool QualityChecker::validateSegmentation(const Image* image, const SegmentationResult& segmentationResult)
{
	//TODO
	return true;
}

/**
 * Checks the quality of the image
 */
bool QualityChecker::checkIrisQuality(const Image* image, const SegmentationResult& segmentationResult)
{
	//TODO
	return true;
}
