#pragma once

#include "irisencoder.h"


#include <memory>

using namespace std;

class LogGabor1DFilter {
public:

	typedef enum { FILTER_REAL, FILTER_IMAG } FilterType;
	LogGabor1DFilter();
	LogGabor1DFilter(double f0, double sigmanOnF, FilterType type=FILTER_IMAG);
	virtual ~LogGabor1DFilter();

	void applyFilter(const Mat_<uint8_t>& image, Mat_<float>& dest, const Mat_<uint8_t>& mask, Mat_<uint8_t>& destMask);

private:
	Mat_< complex<float> > filter;		// Filter in the frequency domain
	Mat_< complex<float> > complexInput;
	Mat_< complex<float> > filterResult;

	double f0, sigmaOnF;
	FilterType type;

	void initializeFilter(const Mat_<uint8_t> image);		// Must release the result
};


class LogGaborEncoder : public IrisEncoder
{
public:
	LogGaborEncoder();
	~LogGaborEncoder();

	static Size getTemplateSize() { return IrisEncoder::getOptimumTemplateSize(256, 20); };

protected:
	Mat_<float> filteredTexture;
	Mat_<uint8_t> filteredMask;
	Mat_<uint8_t> resizedTexture;
	Mat_<uint8_t> resizedMask;
	Mat_<uint8_t> resultTemplate;
	Mat_<uint8_t> resultMask;

	static CvSize getResizedTextureSize();
	vector<LogGabor1DFilter> filterBank;
	virtual IrisTemplate encodeTexture(const Mat_<uint8_t>& texture, const Mat_<uint8_t>& mask);
};
