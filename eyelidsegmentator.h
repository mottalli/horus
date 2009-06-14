/*
 * eyelidsegmentator.h
 *
 *  Created on: Jun 5, 2009
 *      Author: marcelo
 */

#ifndef EYELIDSEGMENTATOR_H_
#define EYELIDSEGMENTATOR_H_

#include "common.h"
#include <cmath>

class EyelidSegmentator {
public:
	EyelidSegmentator();
	virtual ~EyelidSegmentator();

	std::pair<Parabola, Parabola> segmentEyelids(const Image* image, const Circle& pupilCircle, const Circle& irisCircle);

private:
	Parabola segmentUpper(const Image* image, int x0, int y0, int x1, int y1, const Circle& pupilCircle, const Circle& irisCircle);
	Parabola segmentLower(const Image* image, int x0, int y0, int x1, int y1, const Circle& pupilCircle, const Circle& irisCircle);

	std::pair<Parabola, double> findParabola(const Image* image, int p, int x0, int y0, int x1, int y1);
	double parabolaAverage(const Image* gradient, const Image* originalImage, const Parabola& parabola);

	int pupilRadius;
};

#endif /* EYELIDSEGMENTATOR_H_ */
