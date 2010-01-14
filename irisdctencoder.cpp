#include "irisdctencoder.h"


IrisDCTEncoder::IrisDCTEncoder()
{
	this->buffers.normalizedTexture = NULL;
	this->buffers.normalizedNoiseMask = NULL;
}

IrisDCTEncoder::~IrisDCTEncoder()
{
	if (this->buffers.normalizedTexture != NULL) {
		cvReleaseMat(&this->buffers.normalizedTexture);
		cvReleaseMat(&this->buffers.rotatedTexture);
		cvReleaseMat(&this->buffers.normalizedNoiseMask);
		cvReleaseMat(&this->buffers.patch);
		cvReleaseMat(&this->buffers.horizHanningWindow);
		cvReleaseMat(&this->buffers.vertHanningWindow);
		cvReleaseMat(&this->buffers.averagedPatch);
		cvReleaseMat(&this->buffers.patchDCT);
		cvReleaseMat(&this->buffers.dctOutput);
		cvReleaseMat(&this->buffers.codelets);
		cvReleaseMat(&this->buffers.codeletDiff);
	}
}

IrisTemplate IrisDCTEncoder::generateTemplate(const Image* image, const SegmentationResult& segmentationResult)
{
	assert(image->nChannels == 1);
	this->initializeBuffers(image);

	IrisDCTEncoder::normalizeIris(image, this->buffers.normalizedTexture, this->buffers.normalizedNoiseMask, segmentationResult);

	// Mask away pixels too far from the mean
	/*CvScalar smean, sdev;
	cvAvgSdv(this->buffers.normalizedTexture, &smean, &sdev, this->buffers.normalizedNoiseMask);
	double mean = smean.val[0], dev = sdev.val[0];
	uint8_t uthresh = uint8_t(mean+dev);
	uint8_t lthresh = uint8_t(mean-dev);

	for (int y = 0; y < this->buffers.normalizedTexture->height; y++) {
		uint8_t* row = ((uint8_t*)this->buffers.normalizedTexture->imageData) + y*this->buffers.normalizedTexture->widthStep;
		for (int x = 0; x < this->buffers.normalizedTexture->width; x++) {
			uint8_t val = row[x];
			if (val < lthresh || val > uthresh) {
				cvSetReal2D(this->buffers.normalizedNoiseMask, y, x, 0);
			}
		}
	}*/

	this->applyFilter();

	/*CvMat tmp1, tmp2;
	cvGetSubRect(this->buffers.codelets, &tmp1, cvRect(0, 0, 80, this->buffers.codelets->height));
	cvGetSubRect(this->buffers.normalizedNoiseMask, &tmp2, cvRect(0, 0, tmp1.width, tmp1.height));
	return IrisTemplate(&tmp1, &tmp2);*/

	CvMat tmp;
	cvGetSubRect(this->buffers.normalizedNoiseMask, &tmp, cvRect(0, 0, this->buffers.codelets->width, this->buffers.codelets->height));
	return IrisTemplate(this->buffers.codelets, &tmp);

	//return IrisTemplate();
}

void IrisDCTEncoder::applyFilter()
{
	CvMat tmp;
	CvMat tmp2;

	// STEP 1: Rotate the texture by 45 degrees
	this->rotate45(this->buffers.normalizedTexture, this->buffers.rotatedTexture);

	// STEP 2: Extract the horizontal bands
	vector<CvMat*> bands;
	for (int y = 0; y+PATCH_HEIGHT <= ACTUAL_HEIGHT; y += PATCH_HEIGHT_OVERLAP) {
		CvMat* band = cvCreateMat(PATCH_HEIGHT, NORM_WIDTH, CV_8U);
		cvGetSubRect(this->buffers.rotatedTexture, &tmp, cvRect(0, y, NORM_WIDTH, PATCH_HEIGHT));
		cvCopy(&tmp, band);

		bands.push_back(band);
	}

	assert(bands.size() == NUMBER_OF_BANDS);

	// STEP 3: Extract the patches
	assert(bands.size() == 11);			// Paper says it must be 11
	CvMat* patch = this->buffers.patch;
	CvMat* averagedPatch = this->buffers.averagedPatch;

	for (int i = 0; i < NUMBER_OF_BANDS; i++) {
		CvMat* band = bands[i];
		assert(band->height == PATCH_HEIGHT);

		for (int j = 0; j < PATCHES_PER_BAND; j++) {
			int x = j*PATCH_WIDTH_OVERLAP;
			
			cvGetSubRect(band, &tmp, cvRect(x, 0, PATCH_WIDTH, PATCH_HEIGHT));
			cvConvert(&tmp, patch);		// The band is 8 bit pixel while the patch is 32 bit double

			// STEP 4: apply the horizontal Hanning window
			cvMul(this->buffers.horizHanningWindow, patch, patch);

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
			cvMul(averagedPatch, this->buffers.vertHanningWindow, averagedPatch);

			// STEP 7: get the DCT of the patch
			cvDCT(averagedPatch, this->buffers.patchDCT, CV_DXT_FORWARD);

			// Store the output
			cvGetSubRect(this->buffers.dctOutput, &tmp, cvRect(j, i*PATCH_HEIGHT, 1, PATCH_HEIGHT));
			cvCopy(this->buffers.patchDCT, &tmp);
		}
	}

	// STEP 8: calculate the codelets from the differences between the DCT of two consecutive patches
	CvMat codelet0, codelet1;
	for (int i = 0; i < NUMBER_OF_BANDS; i++) {
		for (int j = 0; j < PATCHES_PER_BAND-1; j++) {
			cvGetSubRect(this->buffers.dctOutput, &codelet0, cvRect(j, i*PATCH_HEIGHT, 1, PATCH_HEIGHT));
			cvGetSubRect(this->buffers.dctOutput, &codelet1, cvRect(j+1, i*PATCH_HEIGHT, 1, PATCH_HEIGHT));
			cvSub(&codelet0, &codelet1, this->buffers.codeletDiff);
			cvThreshold(this->buffers.codeletDiff, this->buffers.codeletDiff, 0, 1, CV_THRESH_BINARY);

			cvGetSubRect(this->buffers.codelets, &tmp, cvRect(j, i*DCT_COEFFICIENTS, 1, DCT_COEFFICIENTS));
			cvGetSubRect(this->buffers.codeletDiff, &tmp2, cvRect(0, 0, 1, DCT_COEFFICIENTS));
			//cvCopy(&tmp2, &tmp);
			cvConvert(&tmp2, &tmp);
		}
	}

	// Free up the bands (TODO: make this into buffer)
	for (int i = 0; i < NUMBER_OF_BANDS; i++) {
		cvReleaseMat(&bands[i]);
	}

}

void IrisDCTEncoder::normalizeIris(const Image* image, CvMat* dest, CvMat* destMask, const SegmentationResult& segmentationResult)
{
	int normalizedWidth = dest->width, normalizedHeight = dest->height;
	const Contour& pupilContour = segmentationResult.pupilContour;
	const Contour& irisContour = segmentationResult.irisContour;
	CvPoint p0, p1;

	// Initialize the mask to 1 (all bits enabled)
	cvSet(destMask, cvScalar(1,1,1));

	for (int x = 0; x < normalizedWidth; x++) {
		double theta = (double(x)/double(normalizedWidth)) * (2.0*M_PI);
		double w = (theta/(2.0*M_PI))*double(pupilContour.size());

		p0 = pupilContour[int(std::floor(w))];		p1 = pupilContour[int(std::ceil(w)) % pupilContour.size()];
		double prop = w-std::floor(w);
		double xfrom = double(p0.x) + double(p1.x-p0.x)*prop;
		double yfrom = double(p0.y) + double(p1.y-p0.y)*prop;

		w = (theta/(2.0*M_PI))*double(irisContour.size());
		p0 = irisContour[int(std::floor(w))];
		p1 = irisContour[int(std::ceil(w)) % irisContour.size()];
		prop = w-std::floor(w);
		double xto = double(p0.x) + double(p1.x-p0.x)*prop;
		double yto = double(p0.y) + double(p1.y-p0.y)*prop;

		for (int y = 0; y < normalizedHeight; y++) {
			w = double(y)/double(normalizedHeight-1);
			double ximage = xfrom + w*(xto-xfrom);
			double yimage = yfrom + w*(yto-yfrom);

			int ximage0 = int(std::floor(ximage));
			int ximage1 = int(std::ceil(ximage));
			int yimage0 = int(std::floor(yimage));
			int yimage1 = int(std::ceil(yimage));

			if (ximage0 < 0 || ximage1 >= image->width || yimage0 < 0 || yimage1 >= image->height) {
				cvSetReal2D(dest, y, x, 0);
				cvSetReal2D(destMask, y, x, 0);
			} else {
				double v1 = cvGetReal2D(image, yimage0, ximage0);
				double v2 = cvGetReal2D(image, yimage0, ximage1);
				double v3 = cvGetReal2D(image, yimage1, ximage0);
				double v4 = cvGetReal2D(image, yimage1, ximage1);
				cvSetReal2D(dest, y, x, (v1+v2+v3+v4)/4.0);
			}
		}
	}

	cvDilate(destMask, destMask);
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

void IrisDCTEncoder::initializeBuffers(const Image* image)
{
	Parameters* parameters = Parameters::getParameters();

	if (this->buffers.normalizedTexture == NULL) {
		this->buffers.normalizedTexture = cvCreateMat(NORM_HEIGHT, NORM_WIDTH, CV_8U);
		this->buffers.rotatedTexture = cvCreateMat(NORM_HEIGHT, NORM_WIDTH, CV_8U);
		this->buffers.normalizedNoiseMask = cvCreateMat(NORM_HEIGHT, NORM_WIDTH, CV_8U);
		this->buffers.patch = cvCreateMat(PATCH_HEIGHT, PATCH_WIDTH, CV_32F);
		this->buffers.horizHanningWindow = cvCreateMat(PATCH_HEIGHT, PATCH_WIDTH, CV_32F);
		this->buffers.vertHanningWindow = cvCreateMat(PATCH_HEIGHT, 1, CV_32F);
		this->buffers.averagedPatch = cvCreateMat(PATCH_HEIGHT, 1, CV_32F);
		this->buffers.patchDCT = cvCreateMat(PATCH_HEIGHT, 1, CV_32F);

		this->buffers.dctOutput = cvCreateMat(NUMBER_OF_BANDS * PATCH_HEIGHT, PATCHES_PER_BAND, CV_32F);
		this->buffers.codelets = cvCreateMat(NUMBER_OF_BANDS * DCT_COEFFICIENTS, PATCHES_PER_BAND-1, CV_8U);
		this->buffers.codeletDiff = cvCreateMat(PATCH_HEIGHT, 1, CV_32F);

		// Initialize the "1/4 Hanning window" (it's actually 1/4 in the beginning and 1/4 in the end)
		cvSet(this->buffers.horizHanningWindow, cvScalar(1,1,1));
		double N = double(this->buffers.horizHanningWindow->width);
		for (int i0 = 0; i0 < this->buffers.horizHanningWindow->width/4; i0++) {
			int i1 = i0 + int(0.75*N);
			double v0 = 0.5 * (1.0 - (cos(2.0*M_PI*double(i0) / (N-1))));
			double v1 = 0.5 * (1.0 - (cos(2.0*M_PI*double(i1) / (N-1))));

			for (int j = 0; j < this->buffers.horizHanningWindow->height; j++) {
				cvSetReal2D(this->buffers.horizHanningWindow, j, i0, v0);
				cvSetReal2D(this->buffers.horizHanningWindow, j, i1, v1);
			}
		}

		cvSet(this->buffers.vertHanningWindow, cvScalar(1,1,1));
		N = double(this->buffers.vertHanningWindow->height);
		for (int i0 = 0; i0 < this->buffers.vertHanningWindow->height/4; i0++) {
			int i1 = i0 + int(0.75*N);
			double v0 = 0.5 * (1.0 - (cos(2.0*M_PI*double(i0) / (N-1))));
			double v1 = 0.5 * (1.0 - (cos(2.0*M_PI*double(i1) / (N-1))));

			cvSetReal2D(this->buffers.vertHanningWindow, i0, 0, v0);
			cvSetReal2D(this->buffers.vertHanningWindow, i1, 0, v1);
		}
	}
}
