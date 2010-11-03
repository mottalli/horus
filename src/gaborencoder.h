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

	void applyFilter(const Mat_<float>& src, Mat_<float>& dest, const Mat_<uint8_t>& mask, Mat_<uint8_t>& destMask);

	Mat_<float> filter;

private:
	int width, height;
	float u0, v0, alpha, beta;
	FilterType type;
};

class GaborEncoder : public IrisEncoder
{
public:
	GaborEncoder();
	virtual ~GaborEncoder();

	static Size getTemplateSize() { return IrisEncoder::getOptimumTemplateSize(256, 8); };

protected:
	Mat_<float> filteredTexture;
	Mat_<uint8_t> filteredMask;
	Mat_<float> floatTexture;
	Mat_<uint8_t> resultTemplate;
	Mat_<uint8_t> resultMask;

	virtual IrisTemplate encodeTexture(const Mat_<uint8_t>& texture, const Mat_<uint8_t>& mask);
	std::vector<GaborFilter> filterBank;
};

