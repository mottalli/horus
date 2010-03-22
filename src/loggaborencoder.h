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

	struct {
		IplImage* filter;		// Filter in the frequency domain
	} buffers;

	void applyFilter(const IplImage* image, IplImage* dest, const CvMat* mask, CvMat* destMask);

private:
	double f0, sigmaOnF;
	FilterType type;

	void initializeFilter(const IplImage* image);		// Must release the result
};


class LogGaborEncoder : public IrisEncoder
{
public:
	LogGaborEncoder();
	~LogGaborEncoder();

	IplImage *filteredTexture;
	CvMat *filteredMask;

	static CvSize getTemplateSize() { return IrisEncoder::getOptimumTemplateSize(256, 20); };

protected:
	static CvSize getResizedTextureSize();
	vector<LogGabor1DFilter> filterBank;
	virtual IrisTemplate encodeTexture(const IplImage* texture, const CvMat* mask);
};
