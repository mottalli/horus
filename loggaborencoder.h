#pragma once

#include "irisencoder.h"


#include <memory>

using namespace std;

class LogGabor1DFilter {
public:
	LogGabor1DFilter();
	LogGabor1DFilter(double f0, double sigmanOnF);
	virtual ~LogGabor1DFilter();

	struct {
		IplImage* filter;		// Filter in the frequency domain
	} buffers;

	void applyFilter(const IplImage* image, IplImage* dest, const CvMat* mask, CvMat* destMask);

private:
	double f0, sigmaOnF;

	void initializeFilter(const IplImage* image);		// Must release the result
};


class LogGaborEncoder : public IrisEncoder
{
public:
	LogGaborEncoder();
	~LogGaborEncoder();

	IplImage *filteredTexture, *filteredTextureReal, *filteredTextureImag;
	CvMat *filteredMask;
	CvMat *thresholdedTextureReal, *thresholdedTextureImag;
	CvMat *resultFilter, *resultMask;

protected:
	vector<LogGabor1DFilter> filterBank;
	virtual IrisTemplate encodeTexture(const IplImage* texture, const CvMat* mask);
	void initializeBuffers(const IplImage* texture);
};
