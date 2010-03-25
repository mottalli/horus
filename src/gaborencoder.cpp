#include "gaborencoder.h"

#include "tools.h"

GaborFilter::GaborFilter()
{
}

GaborFilter::GaborFilter(int width, int height, double u0, double v0, double alpha, double beta, FilterType type)
{
	this->width = width;
	this->height = height;
	this->u0 = u0;
	this->v0 = v0;
	this->alpha = alpha;
	this->beta = beta;
	this->type = type;

	double x0 = -1, y0 = -1;
	double x1 = 1, y1 = 1;

	this->filter = cvCreateMat(height, width, CV_32FC1);
	for (int i = 0; i < height; i++) {
		double y = y0 + ((y1-y0)/double(height-1)) * double(i);
		for (int j = 0; j < width; j++) {
			double x = x0 + ((x1-x0)/double(width-1)) * double(j);

			double env = exp(-M_PI* ((x*x)/(alpha*alpha)  + (y*y)/(beta*beta)));		// Gaussian envelope
			double f = 2.0*M_PI*(u0*x + v0*y);

			double carrier;
			if (type == FILTER_REAL) {
				carrier = cos(f);
			} else if (type == FILTER_IMAG) {
				carrier = -sin(f);
			}

			cvSetReal2D(this->filter, i, j, env*carrier);
		}
	}
}

GaborFilter::~GaborFilter()
{
	//cvReleaseMat(&this->filter);
}

void GaborFilter::applyFilter(const CvMat* src, CvMat* dest, const CvMat* mask, CvMat* destMask)
{
	assert(SAME_SIZE(mask, destMask));
	cvFilter2D(src, dest, this->filter);
}

GaborEncoder::GaborEncoder()
{
	this->filterBank.push_back(GaborFilter(15, 15, 0.5, 0.5, 2, 2, GaborFilter::FILTER_IMAG));
	//this->filterBank.push_back(GaborFilter(15, 15, 0.5, -0.5, 2, 2, GaborFilter::FILTER_IMAG));
	//this->filterBank.push_back(GaborFilter(15, 15, 0.5, 1, 2, 2, GaborFilter::FILTER_IMAG));
	//this->filterBank.push_back(GaborFilter(15, 15, -0.5, 1, 2, 2, GaborFilter::FILTER_IMAG));
	this->filteredTexture = NULL;
	this->filteredMask = NULL;
	this->doubleTexture = NULL;
}

GaborEncoder::~GaborEncoder()
{
	if (this->filteredTexture != NULL) {
		cvReleaseMat(&this->filteredTexture);
		cvReleaseMat(&this->filteredMask);
		cvReleaseMat(&this->doubleTexture);
	}
}

IrisTemplate GaborEncoder::encodeTexture(const IplImage* texture, const CvMat* mask)
{
	assert(SAME_SIZE(texture, mask));
	assert(texture->nChannels == 1);
	assert(texture->depth == IPL_DEPTH_8U);

	CvSize templateSize = GaborEncoder::getTemplateSize();
	size_t nFilters = this->filterBank.size();
	size_t nSlots = templateSize.width / nFilters;
	int slotSize = nFilters;

	CvMat* resultTemplate = cvCreateMat(templateSize.height, templateSize.width, CV_8U);
	CvMat* resultMask = cvCreateMat(templateSize.height, templateSize.width, CV_8U);

	Tools::updateSize(&this->doubleTexture, cvGetSize(texture), CV_32F);
	Tools::updateSize(&this->filteredTexture, cvGetSize(texture), CV_32F);
	Tools::updateSize(&this->filteredMask, cvGetSize(texture));

	cvConvert(texture, this->doubleTexture);		// Use double precision


	for (size_t f = 0; f < nFilters; f++) {
		GaborFilter& filter = this->filterBank[f];
		filter.applyFilter(doubleTexture, this->filteredTexture, mask, this->filteredMask);

		/*CvMat* tmp = cvCreateMat(this->filteredTexture->rows, this->filteredTexture->cols, CV_8U);
		cvThreshold(this->filteredTexture, tmp, 0, 1, CV_THRESH_BINARY);
		const CvMat* aMostrar = tmp;
		IplImage* im = cvCreateImage(cvGetSize(aMostrar), IPL_DEPTH_8U, 1);
		cvNormalize(aMostrar, im, 0, 255, CV_MINMAX);
		cvNamedWindow("filtered");
		cvShowImage("filtered", im);
		cvWaitKey(0);*/


		for (size_t s = 0; s < nSlots; s++) {
			int xtemplate = s*slotSize + f;
			int xtexture = (texture->width/nSlots) * s;
			for (int ytemplate = 0; ytemplate < templateSize.height; ytemplate++) {
				int ytexture = (texture->height/templateSize.height) * ytemplate;
				assert(xtexture < this->filteredTexture->width && ytexture < this->filteredTexture->height);

				unsigned char templateBit = (cvGetReal2D(this->filteredTexture, ytexture, xtexture) >  0.0 ? 1 : 0);
				unsigned char maskBit1 = ((cvGetReal2D(this->filteredMask, ytemplate, xtemplate) == 0.0) ? 0 : 1);
				unsigned char maskBit2 = (abs(cvGetReal2D(this->filteredTexture, ytemplate, xtemplate)) < 0.001 ? 0 : 1);

				cvSetReal2D(resultTemplate, ytemplate, xtemplate, templateBit);
				cvSetReal2D(resultMask, ytemplate, xtemplate, maskBit1 & maskBit2);
			}
		}

	}


	IrisTemplate result(resultTemplate, resultMask);

	cvReleaseMat(&resultTemplate);
	cvReleaseMat(&resultMask);

	return result;
}
