/*
 * loggabor1dfilter.h
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#pragma once

#include "common.h"
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

	void applyFilter(const Image* image, Image* dest, CvMat* mask);

private:
	double f0, sigmaOnF;

	void initializeFilter(const Image* image);		// Must release the result
};

