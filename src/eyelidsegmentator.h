#pragma once

#include "common.h"
#include <cmath>

namespace horus {

class EyelidSegmentatorParameters
{
public:
	int parabolicDetectorStep;

	EyelidSegmentatorParameters()
	{
		this->parabolicDetectorStep = 10;
	};
};

class EyelidSegmentator {
public:
	EyelidSegmentator();
	virtual ~EyelidSegmentator();

	std::pair<Parabola, Parabola> segmentEyelids(const GrayscaleImage& image, const Circle& pupilCircle, const Circle& irisCircle);

	EyelidSegmentatorParameters parameters;

private:
	Parabola segmentUpper(const GrayscaleImage& image, const Mat_<float>& gradient, int x0, int y0, int x1, int y1, const Circle& pupilCircle, const Circle& irisCircle);
	Parabola segmentLower(const GrayscaleImage& image, const Mat_<float>& gradient, int x0, int y0, int x1, int y1, const Circle& pupilCircle, const Circle& irisCircle);

	std::pair<Parabola, double> findParabola(const GrayscaleImage& image, const Mat_<float>& gradient, int p, int x0, int y0, int x1, int y1);
	double parabolaAverage(const Mat_<float>& gradient, const GrayscaleImage& originalImage, const Parabola& parabola);

	int pupilRadius;

	Mat_<float> gradient;
};

}
