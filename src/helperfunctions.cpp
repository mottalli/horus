#include "helperfunctions.h"

void HelperFunctions::extractRing(const Mat_<uint8_t>& src, Mat_<uint8_t>& dest, int x0, int y0, int radiusMin, int radiusMax)
{
	assert(src.channels() == 1 && dest.channels() == 1);
	assert(radiusMin < radiusMax);

	int xsrc, ysrc, xdest, ydest;
	double stepRadius = double(radiusMax-radiusMin)/double(dest.rows-1);
	double stepTheta = (2.0*M_PI) / double(dest.cols-1);

	for (ydest = 0; ydest < dest.rows; ydest++) {
		double radius = double(radiusMin) + (stepRadius * double(ydest));
		for (xdest = 0; xdest < dest.cols; xdest++) {
			double theta = stepTheta * double(xdest);

			xsrc = int(double(x0) + radius*cos(theta));
			ysrc = int(double(y0) + radius*sin(theta));

			if (xsrc < 0 || xsrc >= src.cols || ysrc < 0 || ysrc >= src.rows) {
				dest(ydest, xdest) = 0;
			} else {
				dest(ydest, xdest) = src(ysrc, xsrc);
			}
		}
	}
}

void HelperFunctions::smoothSnakeFourier(Mat_<float>& snake, int coefficients)
{
	dft(snake, snake, CV_DXT_FORWARD);
	for (int u = coefficients; u < snake.cols-coefficients; u++) {
		snake(0, u) = 0;
	}
	dft(snake, snake, CV_DXT_INV_SCALE);
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
