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
		Image* filter;		// Filter in the frequency domain
	} buffers;

	void applyFilter(const Image* image, Image* dest, const CvMat* mask, CvMat* destMask);

private:
	double f0, sigmaOnF;

	void initializeFilter(const Image* image);		// Must release the result
};


class LogGaborEncoder : public IrisEncoder
{
public:
	LogGaborEncoder();
protected:
	LogGabor1DFilter filter;
	virtual IrisTemplate encodeTexture(const Image* texture, const CvMat* mask);
};
