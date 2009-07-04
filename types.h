/* 
 * File:   types.h
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:41 PM
 */

#ifndef _TYPES_H
#define	_TYPES_H

#include <vector>
#include <opencv/cv.h>

typedef std::vector<CvPoint> Contour;
typedef IplImage Image;

typedef struct {
    int xc, yc, radius;
} Circle;


typedef std::pair<Contour, Circle> ContourAndCloseCircle;


class Parabola
{
public:
    double x0, y0, p;
    double a, b, c;
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
};

#endif	/* _TYPES_H */

