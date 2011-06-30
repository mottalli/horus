#pragma once

#include "irisencoder.h"

namespace horus {

class LogGabor1DFilter {
public:

	typedef enum { FILTER_REAL, FILTER_IMAG } FilterType;
	LogGabor1DFilter();
	LogGabor1DFilter(double f0, double sigmanOnF, FilterType type=FILTER_IMAG);
	virtual ~LogGabor1DFilter();

	void applyFilter(const GrayscaleImage& image, Mat1d& dest, const GrayscaleImage& mask, Mat1b& destMask) const;

	double f0, sigmaOnF;
	FilterType type;

private:
	mutable Mat_<Complexd> filter;		// Filter in the frequency domain

	static Mat_<Complexd> createRowFilter(Size size, double f0, double sigmaOnF);
};


class LogGaborEncoder : public IrisEncoder
{
public:
	LogGaborEncoder();
	~LogGaborEncoder();

	static Size getTemplateSize() { return IrisEncoder::getOptimumTemplateSize(256, 20); }

	string getEncoderSignature() const;

protected:
	vector<LogGabor1DFilter> filterBank;

	virtual IrisTemplate encodeTexture(const GrayscaleImage& texture, const GrayscaleImage& mask);
	virtual Size getNormalizationSize() const { return LogGaborEncoder::getTemplateSize(); }
};

}
