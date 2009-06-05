/*
 * decorator.h
 *
 *  Created on: Jun 4, 2009
 *      Author: marcelo
 */

#ifndef DECORATOR_H_
#define DECORATOR_H_

#include "common.h"
#include "segmentationresult.h"

class Decorator {
public:
	Decorator();

	void drawSegmentationResult(Image* image, const SegmentationResult& segmentationResult);
private:
	void drawContour(Image* image, const Contour& contour);
	void drawParabola(Image* image, const Parabola& parabola, int xmin, int xmax);
};

#endif /* DECORATOR_H_ */
