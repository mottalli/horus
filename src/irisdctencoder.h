#pragma once
#include "common.h"
#include "iristemplate.h"
#include "segmentationresult.h"
#include "irisencoder.h"

using namespace std;

class IrisDCTEncoder  : public IrisEncoder {
public:
	//static const int NORM_WIDTH = 512;
	// The paper uses 512 but we want the resulting width to be a multiple of 32
	/*static const int NORM_WIDTH = 592;
	static const int NORM_HEIGHT = 80;
	static const int ACTUAL_HEIGHT = 48;
	static const int NUMBER_OF_BANDS = (ACTUAL_HEIGHT / PATCH_HEIGHT_OVERLAP) - 1;
	static const int PATCHES_PER_BAND = (NORM_WIDTH / PATCH_WIDTH_OVERLAP) - 1;*/

	static const int PATCH_WIDTH = 12;
	static const int PATCH_WIDTH_OVERLAP = 6;
	static const int PATCH_HEIGHT = 8;
	static const int PATCH_HEIGHT_OVERLAP = 4;
	static const int DCT_COEFFICIENTS = 4;

	int numberOfBands, patchesPerBand;

	IrisDCTEncoder();
	virtual ~IrisDCTEncoder();

	CvMat* rotatedTexture;
	CvMat* patch;
	CvMat* otherPatch;
	CvMat* averagedPatch;
	CvMat* horizHanningWindow;
	CvMat* vertHanningWindow;
	CvMat* patchDCT;
	CvMat* dctOutput;
	CvMat* codelets;
	CvMat* codeletDiff;

protected:
	virtual IrisTemplate encodeTexture(const IplImage* texture, const CvMat* mask);
	void initializeBuffers(const IplImage* image);
	void applyFilter(const IplImage* texture, const CvMat* mask);

	void rotate45(const CvMat* src, CvMat* dest);
};


