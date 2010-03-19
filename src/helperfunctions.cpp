#include "helperfunctions.h"

void HelperFunctions::extractRing(const IplImage* src, IplImage* dest, int x0, int y0, int radiusMin, int radiusMax)
{
	assert(src->nChannels == 1 && dest->nChannels == 1);
	assert(radiusMin < radiusMax);

	int xsrc, ysrc, xdest, ydest;
	double stepRadius = double(radiusMax-radiusMin)/double(dest->height-1);
	double stepTheta = (2.0*M_PI) / double(dest->width-1);

	for (ydest = 0; ydest < dest->height; ydest++) {
		double radius = double(radiusMin) + (stepRadius * double(ydest));
		for (xdest = 0; xdest < dest->width; xdest++) {
			double theta = stepTheta * double(xdest);

			xsrc = int(double(x0) + radius*cos(theta));
			ysrc = int(double(y0) + radius*sin(theta));

			if (xsrc < 0 || xsrc >= src->width || ysrc < 0 || ysrc >= src->height) {
				cvSetReal2D(dest, ydest, xdest, 0);
			} else {
				cvSetReal2D(dest, ydest, xdest, cvGetReal2D(src, ysrc, xsrc));
			}
		}
	}
}

void HelperFunctions::smoothSnakeFourier(CvMat* snake, int coefficients)
{
	cvDFT(snake, snake, CV_DXT_FORWARD);
	for (int u = coefficients; u < snake->cols-coefficients; u++) {
		cvSetReal2D(snake, 0, u, 0);
	}
	cvDFT(snake, snake, CV_DXT_INV_SCALE);
}

Circle HelperFunctions::approximateCircle(const Contour& contour)
{
	Circle result;

	int n = contour.size();

	int sumX = 0, sumY = 0;
	for (Contour::const_iterator it = contour.begin(); it != contour.end(); it++) {
		sumX += (*it).x;
		sumY += (*it).y;
	}
	result.xc = sumX/n;
	result.yc = sumY/n;

	int bestRadius = 0;
	int x,y;
	for (Contour::const_iterator it = contour.begin(); it != contour.end(); it++) {
        x = (*it).x;
        y = (*it).y;
        if ( (x-result.xc)*(x-result.xc)+(y-result.yc)*(y-result.yc) > bestRadius*bestRadius) {
        	bestRadius = int(sqrt((x-result.xc)*(x-result.xc)+(y-result.yc)*(y-result.yc)));
        }
	}

	result.radius = bestRadius;

	return result;
}