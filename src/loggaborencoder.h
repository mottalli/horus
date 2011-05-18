#pragma once

#include "irisencoder.h"


using namespace std;

class LogGabor1DFilter {
public:

	typedef enum { FILTER_REAL, FILTER_IMAG } FilterType;
	LogGabor1DFilter();
	LogGabor1DFilter(double f0, double sigmanOnF, FilterType type=FILTER_IMAG);
	virtual ~LogGabor1DFilter();

	void applyFilter(const GrayscaleImage& image, Mat1d& dest, const GrayscaleImage& mask, GrayscaleImage& destMask);

	double f0, sigmaOnF;
	FilterType type;

private:
	Mat_<Complexd> filter;		// Filter in the frequency domain
	Mat_<Complexd> complexInput;
	Mat_<Complexd> filterResult;

	void initializeFilter(const GrayscaleImage image);		// Must release the result
};


class LogGaborEncoder : public IrisEncoder
{
public:
	LogGaborEncoder();
	~LogGaborEncoder();

	static Size getTemplateSize() { return IrisEncoder::getOptimumTemplateSize(256, 20); }

	string getEncoderSignature() const;

protected:
	Mat1d filteredTexture;
	GrayscaleImage filteredMask;
	GrayscaleImage resultTemplate;
	GrayscaleImage resultMask;

	vector<LogGabor1DFilter> filterBank;

	virtual IrisTemplate encodeTexture(const GrayscaleImage& texture, const GrayscaleImage& mask);
	virtual Size getNormalizationSize() { return LogGaborEncoder::getTemplateSize(); }
};
