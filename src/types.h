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

namespace horus {

class Contour : public std::vector<Point>
{
public:
	Contour(size_t size) : std::vector<Point>(size) {}
	Contour() : std::vector<Point>() {}

	/**
	 * Given a value between 0 and 1, returns the point the contour in that position (linear interpolation)
	 */
	inline Point2d getPoint(double x) const {
		// assert(0 <= prop && prop <= 1);
		double w = x * double(size());
		size_t i0 = size_t(w) % size();
		size_t i1 = size_t(w) % size();

		if (i0 == i1) {
			return this->at(i0);
		} else {
			// assert(abs(i0-i1) == 1);
			Point2d p0 = this->at(i0), p1 = this->at(i1);
			double prop = w-floor(w);

			double xp = p0.x + (p1.x-p0.x)*prop;
			double yp = p0.y + (p1.y-p0.y)*prop;
			return Point2d(xp, yp);
		}
	}
};

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

}
