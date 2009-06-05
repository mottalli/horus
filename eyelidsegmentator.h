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
	Parabola segmentUpper(const Image* image, int pupilRadius);
	Parabola segmentLower(const Image* image, int pupilRadius);

	std::pair<Parabola, double> findParabola(const Image* image, int p, int yMin, int yMax);
	double parabolaAverage(const Image* gradient, const Image* originalImage, const Parabola& parabola);

	int pupilRadius;
};

#endif /* EYELIDSEGMENTATOR_H_ */
