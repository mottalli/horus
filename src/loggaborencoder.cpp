#include "loggaborencoder.h"
#include "tools.h"

#include <cmath>

LogGabor1DFilter::LogGabor1DFilter()
{
	this->f0 = 1/32.0;
	this->sigmaOnF = 0.5;
	this->buffers.filter = NULL;
}


LogGabor1DFilter::LogGabor1DFilter(double f0, double sigmanOnF, FilterType type):
	f0(f0), sigmaOnF(sigmanOnF), type(type)
{
	this->buffers.filter = NULL;
}

LogGabor1DFilter::~LogGabor1DFilter()
{
	/*if (this->buffers.filter) {
		cvReleaseImage(&this->buffers.filter);
	}*/
}

void LogGabor1DFilter::applyFilter(const IplImage* image, IplImage* dest, const CvMat* mask, CvMat* destMask)
{
	assert(SAME_SIZE(image, dest));
	assert(SAME_SIZE(image, mask));
	assert(SAME_SIZE(mask, destMask));
	assert(dest->nChannels == 1);
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

	if (this->type == FILTER_REAL) {
		cvCopy(resultReal, dest);
	} else if (this->type == FILTER_IMAG) {
		cvCopy(resultImag, dest);
	}

	// Filter out elements with low response to the filter
	IplImage* absResponse = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
	IplImage* tmp = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
	IplImage* responseMask = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	// real^2+imag^2
	cvMul(resultReal, resultReal, tmp);
	cvMul(resultImag, resultImag, absResponse);
	cvAdd(absResponse, tmp, absResponse);

	cvThreshold(absResponse, responseMask, 0.01, 1, CV_THRESH_BINARY);
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
	this->filterBank.push_back(LogGabor1DFilter(1.0/32.0, 0.5, LogGabor1DFilter::FILTER_IMAG));
	this->filterBank.push_back(LogGabor1DFilter(1/16.0, 0.7, LogGabor1DFilter::FILTER_IMAG));
	this->filteredTexture = NULL;
	this->filteredMask = NULL;
}

LogGaborEncoder::~LogGaborEncoder()
{
	if (this->filteredTexture != NULL) {
		cvReleaseImage(&this->filteredTexture);
		cvReleaseMat(&this->filteredMask);
	}
}

IrisTemplate LogGaborEncoder::encodeTexture(const IplImage* texture, const CvMat* mask)
{
	assert(SAME_SIZE(texture, mask));
	assert(texture->nChannels == 1);
	assert(texture->depth == IPL_DEPTH_8U);

	CvSize templateSize = LogGaborEncoder::getTemplateSize();
	CvSize resizedTextureSize = this->getResizedTextureSize();
	size_t nFilters = this->filterBank.size();
	// A slots holds the results of all the filters for a single image pixel, distributed in
	// the horizontal direction.
	size_t nSlots = templateSize.width / nFilters;
	int slotSize = nFilters;

	IplImage* resizedTexture = cvCreateImage(resizedTextureSize, IPL_DEPTH_8U, 1);
	CvMat* resizedMask = cvCreateMat(resizedTextureSize.height, resizedTextureSize.width, CV_8U);
	cvResize(texture, resizedTexture, CV_INTER_LINEAR);
	cvResize(mask, resizedMask, CV_INTER_NN);

	CvMat* resultTemplate = cvCreateMat(templateSize.height, templateSize.width, CV_8U);
	CvMat* resultMask = cvCreateMat(templateSize.height, templateSize.width, CV_8U);

	Tools::updateSize(&this->filteredTexture, resizedTextureSize, IPL_DEPTH_32F);
	Tools::updateSize(&this->filteredMask, resizedTextureSize);


	for (size_t f = 0; f < nFilters; f++) {
		LogGabor1DFilter& filter = this->filterBank[f];
		//filter.applyFilter(texture, this->filteredTexture, mask, this->filteredMask);
		filter.applyFilter(resizedTexture, this->filteredTexture, resizedMask, this->filteredMask);

		for (size_t s = 0; s < nSlots; s++) {
			int xtemplate = s*slotSize + f;
			int xtexture = (resizedTextureSize.width/nSlots) * s;
			for (int ytemplate = 0; ytemplate < templateSize.height; ytemplate++) {
				int ytexture = (resizedTextureSize.height/templateSize.height) * ytemplate;
				assert(xtexture < resizedTextureSize.width && ytexture < resizedTextureSize.height);

				unsigned char templateBit = (cvGetReal2D(this->filteredTexture, ytemplate, xtemplate) > 0 ? 1 : 0);
				unsigned char maskBit = ((cvGetReal2D(this->filteredMask, ytemplate, xtemplate) == 0.0) ? 0 : 1);

				cvSetReal2D(resultTemplate, ytemplate, xtemplate, templateBit);
				//TODO: Fix this, it shouldn't overwrite the mask, it should AND against it
				cvSetReal2D(resultMask, ytemplate, xtemplate, maskBit);
			}
		}
	}

	IrisTemplate result(resultTemplate, resultMask);

	cvReleaseMat(&resultTemplate);
	cvReleaseMat(&resultMask);

	cvReleaseImage(&resizedTexture);
	cvReleaseMat(&resizedMask);

	return result;
}

CvSize LogGaborEncoder::getResizedTextureSize()
{
	return LogGaborEncoder::getTemplateSize();
}

