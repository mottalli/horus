#include "irisdctencoder.h"


IrisDCTEncoder::IrisDCTEncoder()
{
	this->rotatedTexture = NULL;
}

IrisDCTEncoder::~IrisDCTEncoder()
{
	if (this->rotatedTexture != NULL) {
		cvReleaseMat(&this->rotatedTexture);
		cvReleaseMat(&this->patch);
		cvReleaseMat(&this->horizHanningWindow);
		cvReleaseMat(&this->vertHanningWindow);
		cvReleaseMat(&this->averagedPatch);
		cvReleaseMat(&this->patchDCT);
		cvReleaseMat(&this->dctOutput);
		cvReleaseMat(&this->codelets);
		cvReleaseMat(&this->codeletDiff);
	}
}

IrisTemplate IrisDCTEncoder::encodeTexture(const IplImage* texture, const CvMat* mask)
{
	assert(SAME_SIZE(texture, mask));
	assert(texture->nChannels == 1);
	this->initializeBuffers(texture);

	this->applyFilter(texture, mask);

	CvMat tmp;
	cvGetSubRect(mask, &tmp, cvRect(0, 0, this->codelets->width, this->codelets->height));
	return IrisTemplate(this->codelets, &tmp);
}

void IrisDCTEncoder::applyFilter(const IplImage* texture, const CvMat* mask)
{
	CvMat tmp;
	CvMat tmp2;

	// STEP 1: Rotate the texture by 45 degrees
	CvMat matTexture;
	cvGetMat(texture, &matTexture);
	this->rotate45(&matTexture, this->rotatedTexture);

	// STEP 2: Extract the horizontal bands
	vector<CvMat*> bands;
	for (int y = 0; y+PATCH_HEIGHT <= texture->height; y += PATCH_HEIGHT_OVERLAP) {
		CvMat* band = cvCreateMat(PATCH_HEIGHT, texture->width, CV_8U);
		cvGetSubRect(this->rotatedTexture, &tmp, cvRect(0, y, texture->width, PATCH_HEIGHT));
		cvCopy(&tmp, band);

		bands.push_back(band);
	}

	assert(bands.size() == this->numberOfBands);

	// STEP 3: Extract the patches
	CvMat* patch = this->patch;
	CvMat* averagedPatch = this->averagedPatch;

	for (int i = 0; i < this->numberOfBands; i++) {
		CvMat* band = bands[i];
		assert(band->height == PATCH_HEIGHT);

		for (int j = 0; j < this->patchesPerBand; j++) {
			int x = j*PATCH_WIDTH_OVERLAP;
			
			cvGetSubRect(band, &tmp, cvRect(x, 0, PATCH_WIDTH, PATCH_HEIGHT));
			cvConvert(&tmp, patch);		// The band is 8 bit pixel while the patch is 32 bit double

			// STEP 4: apply the horizontal Hanning window
			cvMul(this->horizHanningWindow, patch, patch);

			// STEP 5: average the patch
			for (int y = 0; y < PATCH_HEIGHT; y++) {
				double mean = 0;
				for (int x = 0; x < PATCH_WIDTH; x++) {
					mean += cvGetReal2D(patch, y, x);
				}
				mean = mean / double(PATCH_WIDTH);
				cvSetReal2D(averagedPatch, y, 0, mean);
			}

			// STEP 6: apply the vertical Hanning window
			cvMul(averagedPatch, this->vertHanningWindow, averagedPatch);

			// STEP 7: get the DCT of the patch
			cvDCT(averagedPatch, this->patchDCT, CV_DXT_FORWARD);

			// Store the output
			cvGetSubRect(this->dctOutput, &tmp, cvRect(j, i*PATCH_HEIGHT, 1, PATCH_HEIGHT));
			cvCopy(this->patchDCT, &tmp);
		}
	}

	// STEP 8: calculate the codelets from the differences between the DCT of two consecutive patches
	CvMat codelet0, codelet1;
	for (int i = 0; i < this->numberOfBands; i++) {
		for (int j = 0; j < this->patchesPerBand-1; j++) {
			cvGetSubRect(this->dctOutput, &codelet0, cvRect(j, i*PATCH_HEIGHT, 1, PATCH_HEIGHT));
			cvGetSubRect(this->dctOutput, &codelet1, cvRect(j+1, i*PATCH_HEIGHT, 1, PATCH_HEIGHT));
			cvSub(&codelet0, &codelet1, this->codeletDiff);
			cvThreshold(this->codeletDiff, this->codeletDiff, 0, 1, CV_THRESH_BINARY);

			cvGetSubRect(this->codelets, &tmp, cvRect(j, i*DCT_COEFFICIENTS, 1, DCT_COEFFICIENTS));
			cvGetSubRect(this->codeletDiff, &tmp2, cvRect(0, 0, 1, DCT_COEFFICIENTS));
			//cvCopy(&tmp2, &tmp);
			cvConvert(&tmp2, &tmp);
		}
	}

	// Free up the bands (TODO: make this into buffer)
	for (int i = 0; i < this->numberOfBands; i++) {
		cvReleaseMat(&bands[i]);
	}

}

void IrisDCTEncoder::rotate45(const CvMat* src, CvMat* dest)
{
	assert(SAME_SIZE(src, dest));

	CvMat pieceSrc, pieceDest;

	for (int y = 1; y < src->height; y++) {
		int step = y;
		cvGetSubRect(src, &pieceSrc, cvRect(0, y, src->width-step, 1));
		cvGetSubRect(dest, &pieceDest, cvRect(step, y, src->width-step, 1));
		cvCopy(&pieceSrc, &pieceDest);

		cvGetSubRect(src, &pieceSrc, cvRect(src->width-step, y, step, 1));
		cvGetSubRect(dest, &pieceDest, cvRect(0, y, step, 1));
		cvCopy(&pieceSrc, &pieceDest);
	}
}

void IrisDCTEncoder::initializeBuffers(const IplImage* image)
{
	Parameters* parameters = Parameters::getParameters();

	this->numberOfBands = (image->height / PATCH_HEIGHT_OVERLAP) - 1;
	this->patchesPerBand = (image->width / PATCH_WIDTH_OVERLAP) - 1;

	if (this->rotatedTexture == NULL || this->rotatedTexture->width != image->width || this->rotatedTexture->height != image->height) {
		this->rotatedTexture = cvCreateMat(image->height, image->width, CV_8U);
		this->patch = cvCreateMat(PATCH_HEIGHT, PATCH_WIDTH, CV_32F);
		this->horizHanningWindow = cvCreateMat(PATCH_HEIGHT, PATCH_WIDTH, CV_32F);
		this->vertHanningWindow = cvCreateMat(PATCH_HEIGHT, 1, CV_32F);
		this->averagedPatch = cvCreateMat(PATCH_HEIGHT, 1, CV_32F);
		this->patchDCT = cvCreateMat(PATCH_HEIGHT, 1, CV_32F);

		this->dctOutput = cvCreateMat(this->numberOfBands * PATCH_HEIGHT, this->patchesPerBand, CV_32F);
		this->codelets = cvCreateMat(this->numberOfBands * DCT_COEFFICIENTS, this->patchesPerBand-1, CV_8U);
		this->codeletDiff = cvCreateMat(PATCH_HEIGHT, 1, CV_32F);

		// Initialize the "1/4 Hanning window" (it's actually 1/4 in the beginning and 1/4 in the end)
		cvSet(this->horizHanningWindow, cvScalar(1,1,1));
		double N = double(this->horizHanningWindow->width);
		for (int i0 = 0; i0 < this->horizHanningWindow->width/4; i0++) {
			int i1 = i0 + int(0.75*N);
			double v0 = 0.5 * (1.0 - (cos(2.0*M_PI*double(i0) / (N-1))));
			double v1 = 0.5 * (1.0 - (cos(2.0*M_PI*double(i1) / (N-1))));

			for (int j = 0; j < this->horizHanningWindow->height; j++) {
				cvSetReal2D(this->horizHanningWindow, j, i0, v0);
				cvSetReal2D(this->horizHanningWindow, j, i1, v1);
			}
		}

		cvSet(this->vertHanningWindow, cvScalar(1,1,1));
		N = double(this->vertHanningWindow->height);
		for (int i0 = 0; i0 < this->vertHanningWindow->height/4; i0++) {
			int i1 = i0 + int(0.75*N);
			double v0 = 0.5 * (1.0 - (cos(2.0*M_PI*double(i0) / (N-1))));
			double v1 = 0.5 * (1.0 - (cos(2.0*M_PI*double(i1) / (N-1))));

			cvSetReal2D(this->vertHanningWindow, i0, 0, v0);
			cvSetReal2D(this->vertHanningWindow, i1, 0, v1);
		}
	}
}
