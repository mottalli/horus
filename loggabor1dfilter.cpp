/*
 * loggabor1dfilter.cpp
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#include "loggabor1dfilter.h"
#include <cmath>
#include <iostream>

LogGabor1DFilter::LogGabor1DFilter()
{
	this->f0 = 1/32.0;
	this->sigmaOnF = 0.5;
	this->buffers.filter = NULL;
}


LogGabor1DFilter::LogGabor1DFilter(double f0, double sigmanOnF):
	f0(f0), sigmaOnF(sigmanOnF)
{
	this->buffers.filter = NULL;
}

LogGabor1DFilter::~LogGabor1DFilter()
{
	if (this->buffers.filter) {
		cvReleaseImage(&this->buffers.filter);
	}
}

void LogGabor1DFilter::applyFilter(const Image* image, Image* dest, CvMat* mask)
{
	assert(SAME_SIZE(image, dest));
	assert(SAME_SIZE(image, mask));
	assert(dest->nChannels == 2);
	assert(dest->depth == IPL_DEPTH_32F);

	this->initializeFilter(image);

	Image* src = cvCloneImage(image);
	cvSmooth(src, src, CV_GAUSSIAN, 3);

	IplImage* lineReal = cvCreateImage(cvSize(image->width, 1), IPL_DEPTH_32F, 1);
	IplImage* lineImag = cvCreateImage(cvSize(image->width, 1), IPL_DEPTH_32F, 1);
	IplImage* line = cvCreateImage(cvSize(image->width, 1), IPL_DEPTH_32F, 2);

	IplImage* resultReal = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
	IplImage* resultImag = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);

	for (int y = 0; y < image->height; y++) {
		CvMat imline;
		cvGetSubRect(src, &imline, cvRect(0, y, src->width, 1));
		cvConvert(&imline, lineReal);
		cvZero(lineImag);
		cvMerge(lineReal, lineImag, NULL, NULL, line);
		cvDFT(line, line, CV_DXT_FORWARD);

		cvMulSpectrums(line, this->buffers.filter, line, 0);
		cvDFT(line, line, CV_DXT_INV_SCALE);
		cvSplit(line, lineReal, lineImag, NULL, NULL);

		cvGetSubRect(resultReal, &imline, cvRect(0, y, image->width, 1));
		cvCopy(lineReal, &imline);
		cvGetSubRect(resultImag, &imline, cvRect(0, y, image->width, 1));
		cvCopy(lineImag, &imline);

	}

	cvMerge(resultReal, resultImag, NULL, NULL, dest);

	// Filter out elements with low response to the filter
	IplImage* absResponse = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
	IplImage* tmp = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
	IplImage* responseMask = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	// real^2+imag^2
	cvMul(resultReal, resultReal, tmp);
	cvMul(resultImag, resultImag, absResponse);
	cvAdd(absResponse, tmp, absResponse);

	cvThreshold(absResponse, responseMask, 0.001, 1, CV_THRESH_BINARY);
	// Add the bits to the mask
	cvAnd(mask, responseMask, mask);

	cvReleaseImage(&absResponse);
	cvReleaseImage(&tmp);
	cvReleaseImage(&responseMask);

	cvReleaseImage(&src);
	cvReleaseImage(&lineReal);
	cvReleaseImage(&lineImag);
	cvReleaseImage(&line);
	cvReleaseImage(&resultReal);
	cvReleaseImage(&resultImag);
}

void LogGabor1DFilter::initializeFilter(const Image* image)
{
	if (this->buffers.filter== NULL || this->buffers.filter->width != image->width) {
		if (this->buffers.filter != NULL) {
			cvReleaseImage(&this->buffers.filter);
		}

		this->buffers.filter = cvCreateImage(cvSize(image->width, 1), IPL_DEPTH_32F, 2);

		// Re-create the filter
		IplImage* realPart = cvCreateImage(cvSize(image->width, 1), IPL_DEPTH_32F, 1);
		IplImage* imagPart = cvCreateImage(cvSize(image->width, 1), IPL_DEPTH_32F, 1);

		cvZero(imagPart);

		double q = 2.0*log(this->sigmaOnF)*log(this->sigmaOnF);
		cvSetReal2D(realPart, 0, 0, 0.0);

		for (double i = 0; i < image->width; i++) {
			double r = (0.5/double(image->width-1)) * double(i);
			double value = exp(-(log(r/f0)*log(r/f0))/q);
			cvSetReal2D(realPart, 0, i, value);
		}

		cvMerge(realPart, imagPart, NULL, NULL, this->buffers.filter);

		cvReleaseImage(&realPart);
		cvReleaseImage(&imagPart);
	}
}
