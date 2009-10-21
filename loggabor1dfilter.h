/*
 * loggabor1dfilter.h
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#ifndef LOGGABOR1DFILTER_H_
#define LOGGABOR1DFILTER_H_

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

	void applyFilter(const Image* image, Image* dest);

private:
	double f0, sigmaOnF;

	void initializeFilter(const Image* image);		// Must release the result
};

#endif /* LOGGABOR1DFILTER_H_ */
