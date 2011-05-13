#pragma once

#include "common.h"
#include "irisencoder.h"

class GaborFilter
{
public:
	typedef enum { FILTER_REAL, FILTER_IMAG } FilterType;
	GaborFilter();
	GaborFilter(int width, int height, float u0, float v0, float alpha, float beta, FilterType type=FILTER_IMAG);
	virtual ~GaborFilter();

	void applyFilter(const Mat_<float>& src, Mat_<float>& dest, const GrayscaleImage& mask, GrayscaleImage& destMask);

	Mat_<float> filter;

	int width, height;
	float u0, v0, alpha, beta;
	FilterType type;

private:
};

class GaborEncoder : public IrisEncoder
{
public:
	GaborEncoder();
	virtual ~GaborEncoder();

	static Size getTemplateSize() { return IrisEncoder::getOptimumTemplateSize(256, 20); }

	string getEncoderSignature() const;

protected:
	Mat_<float> filteredTexture;
	GrayscaleImage filteredMask;
	Mat_<float> floatTexture;
	GrayscaleImage resultTemplate;
	GrayscaleImage resultMask;

	virtual IrisTemplate encodeTexture(const GrayscaleImage& texture, const GrayscaleImage& mask);
	std::vector<GaborFilter> filterBank;
};

