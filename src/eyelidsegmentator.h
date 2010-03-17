/*
 * eyelidsegmentator.h
 *
 *  Created on: Jun 5, 2009
 *      Author: marcelo
 */

#pragma once

#include "common.h"
#include <cmath>

class EyelidSegmentator {
public:
	EyelidSegmentator();
	virtual ~EyelidSegmentator();

	std::pair<Parabola, Parabola> segmentEyelids(const IplImage* image, const Circle& pupilCircle, const Circle& irisCircle);

private:
	Parabola segmentUpper(const IplImage* image, const IplImage* gradient, int x0, int y0, int x1, int y1, const Circle& pupilCircle, const Circle& irisCircle);
	Parabola segmentLower(const IplImage* image, const IplImage* gradient, int x0, int y0, int x1, int y1, const Circle& pupilCircle, const Circle& irisCircle);

	std::pair<Parabola, double> findParabola(const IplImage* image, const IplImage* gradient, int p, int x0, int y0, int x1, int y1);
	double parabolaAverage(const IplImage* gradient, const IplImage* originalImage, const Parabola& parabola);

	int pupilRadius;
};

