#pragma once

#include "common.h"
#include "irisencoder.h"

class GaborFilter
{
public:
	typedef enum { FILTER_REAL, FILTER_IMAG } FilterType;
	GaborFilter();
	GaborFilter(int width, int height, double u0, double v0, double alpha, double beta, FilterType type=FILTER_IMAG);
	virtual ~GaborFilter();

	void applyFilter(const CvMat* src, CvMat* dest, const CvMat* mask, CvMat* destMask);

	CvMat* filter;

private:
	int width, height;
	double u0, v0, alpha, beta;
	FilterType type;
};

class GaborEncoder : public IrisEncoder
{
public:
	GaborEncoder();
	virtual ~GaborEncoder();

	CvMat *filteredTexture;
	CvMat *filteredMask;
	CvMat *doubleTexture;

	static CvSize getTemplateSize() { return IrisEncoder::getOptimumTemplateSize(256, 8); };

protected:
	virtual IrisTemplate encodeTexture(const IplImage* texture, const CvMat* mask);
	std::vector<GaborFilter> filterBank;
};

