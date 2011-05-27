#include "irissegmentator.h"
#include "tools.h"
#include <cmath>

IrisSegmentator::IrisSegmentator()
{
}


IrisSegmentator::~IrisSegmentator() {
}

ContourAndCloseCircle IrisSegmentator::segmentIris(const GrayscaleImage& image, const ContourAndCloseCircle& pupilSegmentation)
{
	assert(image.channels() == 1 && image.depth() == CV_8U);
	return this->segmentIrisRecursive((const GrayscaleImage&)image, pupilSegmentation, -1, -1);
}

ContourAndCloseCircle IrisSegmentator::segmentIrisRecursive(const GrayscaleImage& image, const ContourAndCloseCircle& pupilSegmentation, int radiusMax, int radiusMin)
{
	this->setupBuffers(image);

	Circle pupilCircle = pupilSegmentation.second;
	if (radiusMin < 0) {
	    radiusMin = pupilCircle.radius * 1.3;
	}

	if (radiusMax < 0) {
	    radiusMax = pupilCircle.radius * 5.0;
	}

	Tools::extractRing(image, this->adjustmentRing, pupilCircle.center.x, pupilCircle.center.y, radiusMin, radiusMax);

	blur(this->adjustmentRing, this->adjustmentRing, Size(3, 7));
	Sobel(this->adjustmentRing, this->adjustmentRingGradient, CV_16S, 0, 1, 3);

	const double theta0 = -M_PI/4.0;
	//const double theta1 = M_PI/4.0;
	//const double theta2 = 3.0*M_PI/4.0;
	const double theta1 = 3.0*M_PI/8.0;
	const double theta2 = 5.0*M_PI/8.0;
	const double theta3 = 5.0*M_PI/4.0;

	Mat_<int16_t>& gradient = this->adjustmentRingGradient;
	Mat_<float>& snake = this->adjustmentSnake;

	assert(snake.cols == this->adjustmentRing.cols);

	int x0 = int((theta0/(2.0*M_PI))*double(snake.cols));
	int x1 = int((theta1/(2.0*M_PI))*double(snake.cols));
	int x2 = int((theta2/(2.0*M_PI))*double(snake.cols));
	int x3 = int((theta3/(2.0*M_PI))*double(snake.cols));

	assert((x1 * x2 * x3) > 0);		// x1, x2 and x3 must be positive (x0 may be negative)
	#define XIMAGE(x) ((x) >= 0 ? (x) : snake.cols+(x))

	int sumY1, maxSumY1 = INT_MIN, sumY2, maxSumY2 = INT_MIN;
	int bestY1 = 0, bestY2 = 0;

	for (int y = 0; y < gradient.rows; y++) {
		sumY1 = 0;
		sumY2 = 0;

		//int16_t* row = (int16_t*)(gradient->imageData + y*gradient->widthStep);
		const int16_t* row = gradient.ptr<int16_t>(y);

		for (int x = x0; x < x1; x++) {
			sumY1 += row[XIMAGE(x)];
		}
		for (int x = x2; x < x3; x++) {
			sumY2 += row[x];		// No need to use XIMAGE here
		}

		if (sumY1 > maxSumY1) {
			maxSumY1 = sumY1;
			bestY1 = y;
		}
		if (sumY2 > maxSumY2) {
			maxSumY2 = sumY2;
			bestY2 = y;
		}
	}

	// If there's a big difference between bestY1 and bestY2, it probably means something wasn't right.
	if (abs(bestY1-bestY2) > gradient.rows*0.2) {
		// Try again with a smaller radiusMax (less chances of error)
		radiusMax = radiusMax - pupilCircle.radius/2;
		if (radiusMax > radiusMin) {
			return this->segmentIrisRecursive(image, pupilSegmentation, radiusMax);
		}
	}


	for (int x = x0; x < x1; x++) {
		snake(0, XIMAGE(x)) = bestY1;
	}

	for (int x = x2; x < x3; x++) {
		snake(0, x) = bestY2;
	}

	// Interpolation between the two segments
	for (int x = x1; x < x2; x++) {
		int y = bestY1 + (double(x-x1)/double(x2-1-x1)) * double(bestY2-bestY1);
		snake(0, x) = y;
	}
	for (int x = x3; x < XIMAGE(x0) && x < snake.cols; x++) {
		//int y = bestY2 + (double(x-x3)/double(snake.cols-1-x3)) * double(bestY1-bestY2);
		int y = bestY2 + (double(x-x3)/double(XIMAGE(x0)-1-x3)) * double(bestY1-bestY2);
		snake(0, x) = y;
	}

	for (int x = 0; x < snake.cols; x++) {
		int maxGrad = INT_MIN;
		int maxY = 0;
		int v = snake(0, x);
		int y0 = max(0, v-5);
		int y1 = min(gradient.rows, v+5);

		for (int y = y0; y < y1; y++) {
			int gxy = gradient(y, x);
			if (gxy > maxGrad) {
				maxGrad = gxy;
				maxY = y;
			}
		}
		snake(0, x) = maxY;
	}

	// Smooth the snake
	Tools::smoothSnakeFourier(snake, 3);

	// Convert to image coordinates
	Contour irisContour(snake.cols);
	for (int x = 0; x < snake.cols; x++) {
		double theta = (double(x)/double(snake.cols))*2.0*M_PI;
		double radius = ((double(snake(0,x))/double(gradient.rows-1))*double(radiusMax-radiusMin)) + double(radiusMin);
		int ximage = int(double(pupilCircle.center.x) + cos(theta) * radius);
		int yimage = int(double(pupilCircle.center.y) + sin(theta) * radius);
		irisContour[x] = Point(ximage, yimage);
	}

	ContourAndCloseCircle result;

	result.first = irisContour;
	result.second = Tools::approximateCircle(result.first);

    return result;
}

void IrisSegmentator::setupBuffers(const GrayscaleImage&)
{
	this->adjustmentSnake.create(1, this->parameters.irisAdjustmentRingWidth);
	this->adjustmentRing.create(Size(this->parameters.irisAdjustmentRingWidth, this->parameters.irisAdjustmentRingHeight));
}
