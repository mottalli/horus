/*
 * File:   helperfunctions.h
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:16 PM
 */

#pragma once

#include "common.h"

namespace HelperFunctions
{
	void extractRing(const IplImage* src, IplImage* dest, int x0, int y0, int radiusMin, int radiusMax);
	void smoothSnakeFourier(CvMat* snake, int coefficients);
	Circle approximateCircle(const Contour& contour);
}



