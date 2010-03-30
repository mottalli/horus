#pragma once

#include "common.h"

namespace HelperFunctions
{
	void extractRing(const IplImage* src, IplImage* dest, int x0, int y0, int radiusMin, int radiusMax);
	void extractRing(const Mat_<uint8_t>& src, Mat_<uint8_t>& dest, int x0, int y0, int radiusMin, int radiusMax);
	void smoothSnakeFourier(CvMat* snake, int coefficients);
	void smoothSnakeFourier(Mat_<float>& snake, int coefficients);
	Circle approximateCircle(const Contour& contour);
}



