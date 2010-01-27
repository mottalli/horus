/*
 * File:   helperfunctions.h
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:16 PM
 */

#pragma once

#include <cmath>
#include <assert.h>
#include "common.h"

namespace HelperFunctions
{
	void extractRing(const Image* src, Image* dest, int x0, int y0, int radiusMin, int radiusMax);
	void smoothSnakeFourier(CvMat* snake, int coefficients);
	Circle approximateCircle(const Contour& contour);
}



