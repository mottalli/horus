/* 
 * File:   types.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:41 PM
 */

#pragma once

#include <vector>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <stdint.h>

using namespace cv;

typedef vector<Point> Contour;

typedef struct {
	Point center;
	int radius;
	inline double perimeter() const { return 2.0*M_PI*double(radius); }
} Circle;


typedef std::pair<Contour, Circle> ContourAndCloseCircle;
typedef Mat_<uint8_t> GrayscaleImage;
typedef Mat_<Vec3b> ColorImage;
typedef Mat Image;									// Either grayscale or color image

class Parabola
{
public:
	double x0, y0, p;

	Parabola()
	{ x0 = y0 = p = a = b = c = 0.0; }

	~Parabola() {}

	Parabola(double x0_, double y0_, double p_)
	{
		x0 = x0_; y0 = y0_; p = p_;
		a = (1.0/(4.0*p)); b = -2.0*a*x0; c = a*(x0*x0+4.0*p*y0);
	}

	inline double value(double x) const
	{
		return a*x*x+b*x+c;
	}
private:
	double a, b, c;
};

struct SegmentationResult {
	Contour irisContour;
	Contour pupilContour;
	Circle pupilCircle;
	Circle irisCircle;
	Parabola upperEyelid;
	Parabola lowerEyelid;

	double pupilContourQuality;

	bool eyelidsSegmented;
};
