/*
 * File:   helperfunctions.h
 * Author: marcelo
 *
 * Created on January 22, 2009, 2:16 PM
 */

#ifndef _HELPERFUNCTIONS_H
#define	_HELPERFUNCTIONS_H

#include <cmath>
#include <assert.h>
#include "common.h"

namespace HelperFunctions
{
	void extractRing(const Image* src, Image* dest, const Circle& centerCircle, double widthFactor);
	void smoothSnakeFourier(CvMat* snake, int coefficients);
}


#endif	/* _HELPERFUNCTIONS_H */

