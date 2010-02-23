#include "loggaborencoder.h"

#include <cmath>

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

void LogGabor1DFilter::applyFilter(const IplImage* image, IplImage* dest, const CvMat* mask, CvMat* destMask)
{
	assert(SAME_SIZE(image, dest));
	assert(SAME_SIZE(image, mask));
	assert(SAME_SIZE(mask, destMask));
	assert(dest->nChannels == 2);
	assert(dest->depth == IPL_DEPTH_32F);

	this->initializeFilter(image);

	IplImage* src = cvCloneImage(image);
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
	cvAnd(mask, responseMask, destMask);

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

void LogGabor1DFilter::initializeFilter(const IplImage* image)
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

LogGaborEncoder::LogGaborEncoder()
{
	this->filterBank.push_back(LogGabor1DFilter(1.0/32.0, 0.5));
	this->filteredTexture = NULL;
}

LogGaborEncoder::~LogGaborEncoder()
{
	if (this->filteredTexture != NULL) {
		cvReleaseImage(&this->filteredTexture);
		cvReleaseImage(&this->filteredTextureReal);
		cvReleaseImage(&this->filteredTextureImag);
		cvReleaseMat(&this->filteredMask);
		cvReleaseMat(&this->thresholdedTextureReal);
		cvReleaseMat(&this->thresholdedTextureImag);
		cvReleaseMat(&this->resultFilter);
		cvReleaseMat(&this->resultMask);
	}
}

IrisTemplate LogGaborEncoder::encodeTexture(const Image* texture, const CvMat* mask)
{
	assert(SAME_SIZE(texture, mask));
	this->initializeBuffers(texture);

	size_t n = this->filterBank.size();
	CvMat dest;

	for (size_t i = 0; i < n; i++) {
		LogGabor1DFilter& filter = this->filterBank[i];
		filter.applyFilter(texture, this->filteredTexture, mask, this->filteredMask);

		cvSplit(this->filteredTexture, this->filteredTextureReal, this->filteredTextureImag, NULL, NULL);
		cvThreshold(this->filteredTextureReal, this->thresholdedTextureReal, 0, 1, CV_THRESH_BINARY);
		cvThreshold(this->filteredTextureImag, this->thresholdedTextureImag, 0, 1, CV_THRESH_BINARY);

		// Merge the result of this filter with the final result
		cvGetSubRect(this->resultFilter, &dest, cvRect(0, i*texture->height, texture->width, texture->height));
		cvCopy(this->thresholdedTextureReal, &dest);
		cvGetSubRect(this->resultMask, &dest, cvRect(0, i*texture->height, texture->width, texture->height));
		cvCopy(this->filteredMask, &dest);
	}

	IrisTemplate result(this->resultFilter, this->resultMask);
	return result;
}

void LogGaborEncoder::initializeBuffers(const IplImage* texture)
{
	if (!this->filteredTexture || (this->filteredTexture->width != texture->width || this->filteredTexture->height != texture->height)) {
		if (this->filteredTexture != NULL) {
			cvReleaseImage(&this->filteredTexture);
			cvReleaseImage(&this->filteredTextureReal);
			cvReleaseImage(&this->filteredTextureImag);
			cvReleaseMat(&this->filteredMask);
			cvReleaseMat(&this->thresholdedTextureReal);
			cvReleaseMat(&this->thresholdedTextureImag);
			cvReleaseMat(&this->resultFilter);
			cvReleaseMat(&this->resultMask);
		}

		this->filteredTexture = cvCreateImage(cvGetSize(texture), IPL_DEPTH_32F, 2);
		this->filteredMask = cvCreateMat(texture->height, texture->width, CV_8U);
		this->filteredTextureReal = cvCreateImage(cvGetSize(texture), IPL_DEPTH_32F, 1);
		this->filteredTextureImag = cvCreateImage(cvGetSize(texture), IPL_DEPTH_32F, 1);
		this->thresholdedTextureReal = cvCreateMat(texture->height, texture->width, CV_8U);
		this->thresholdedTextureImag = cvCreateMat(texture->height, texture->width, CV_8U);

		size_t n = this->filterBank.size();
		this->resultFilter = cvCreateMat(n*texture->height, texture->width, CV_8U);
		this->resultMask = cvCreateMat(n*texture->height, texture->width, CV_8U);
	}
}
