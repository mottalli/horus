/*
 * File:   pupilsegmentator.cpp
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:39 PM
 */

#include "pupilsegmentator.h"
#include "tools.h"
#include "tools.h"

using namespace horus;

PupilSegmentator::PupilSegmentator()
{
	this->_lastSigma = this->_lastMu = -100.0;
	uchar strel[] =  { 0, 0, 0, 0, 0, 0, \
					  0, 0, 255, 255, 0, 0, \
					  0, 255, 255, 255, 255, 0, \
					  0, 255, 255, 255, 255, 0, \
					  0, 0, 255, 255, 0, 0, \
					  0, 0, 0, 0, 0, 0, \
					};
	this->matStructElem = GrayscaleImage(6, 6, strel).clone();
}

PupilSegmentator::~PupilSegmentator()
{
}

ContourAndCloseCircle PupilSegmentator::segmentPupil(const GrayscaleImage& image)
{
	assert(image.channels() == 1 && image.depth() == CV_8U);

	this->setupBuffers(image);
	ContourAndCloseCircle result;

	Circle pupilCircle = this->approximatePupil(this->workingImage);

	pupilCircle.radius /= this->resizeFactor;
	pupilCircle.center.x /= this->resizeFactor;
	pupilCircle.center.y /= this->resizeFactor;

	result.first = this->adjustPupilContour(image, pupilCircle);
	result.second = tools::approximateCircle(result.first);

	return result;

}

void PupilSegmentator::setupBuffers(const Image& image)
{
	// Initialize the working image
	int bufferWidth = this->parameters.bufferWidth;
	int width = image.cols;

	if (width <= bufferWidth) {
		this->resizeFactor = 1.0;
		this->workingImage = image;
	} else {
		this->resizeFactor = double(bufferWidth) / double(width);
		resize(image, this->workingImage, Size(), this->resizeFactor, this->resizeFactor);
	}


	this->adjustmentRing.create(Size(this->parameters.pupilAdjustmentRingWidth, this->parameters.pupilAdjustmentRingHeight));
	this->adjustmentRingGradient.create(Size(this->parameters.pupilAdjustmentRingWidth, this->parameters.pupilAdjustmentRingHeight));
	this->adjustmentSnake.create(Size(this->parameters.pupilAdjustmentRingWidth, 1));
}

Circle PupilSegmentator::approximatePupil(const GrayscaleImage& image)
{
	// First, equalize the image and apply the similarity transform
	//equalizeHist(image, this->equalizedImage);
	tools::stretchHistogram(image, this->equalizedImage, 0.01, 0.0);
	this->similarityTransform();
	blur(this->similarityImage, this->similarityImage, Size(7, 7));

	if (this->parameters.avoidPupilReflection) {
		morphologyEx(this->similarityImage, this->similarityImage, MORPH_DILATE, matStructElem, Point(-1,-1), 4);
	}

	// Now perform the cascaded integro-differential operator (use the ROI if any)
	Circle res;
	Rect ROI = this->eyeROI;
	ROI.x /= this->resizeFactor;
	ROI.y /= this->resizeFactor;
	ROI.width /= this->resizeFactor;
	ROI.height /= this->resizeFactor;
	res = this->cascadedIntegroDifferentialOperator(this->similarityImage, ROI);

	return res;
}

Contour PupilSegmentator::adjustPupilContour(const GrayscaleImage& image, const Circle& approximateCircle)
{
	int radiusMin = approximateCircle.radius * 0.5, radiusMax =
			approximateCircle.radius * 1.5;
	tools::extractRing(image, this->adjustmentRing,
			approximateCircle.center.x, approximateCircle.center.y, radiusMin, radiusMax);

	int infraredThreshold = this->parameters.infraredThreshold;


	// Try to find the region where the infrarred light is
	int x0=INT_MAX, x1=INT_MIN;									// Right / left limit for the infrarred light
	for (int y = 0; y < this->adjustmentRing.rows; y++) {
		const uint8_t* ptr = this->adjustmentRing.ptr(y);
		for (int x = 0; x < this->adjustmentRing.cols/2; x++) {
			if (ptr[x] > infraredThreshold) {
				x0 = min(x0, x);
			}
			if (ptr[this->adjustmentRing.cols/2-x] > infraredThreshold) {
				x1 = max(x1, this->adjustmentRing.cols/2-x);
			}
		}
	}

	bool hasInfrarred = (x0 < INT_MAX && x1 > INT_MIN);

	if (hasInfrarred) {
		x0 = max(1, x0-int(this->adjustmentRing.cols*0.1));
		x1 = min(this->adjustmentRing.cols-2, x1+int(this->adjustmentRing.cols*0.1));
	}


	// Calculate the vertical gradient
	Sobel(this->adjustmentRing, this->adjustmentRingGradient, this->adjustmentRingGradient.type(), 0, 1, 3);
	blur(this->adjustmentRingGradient, this->adjustmentRingGradient, Size(3,3));

	// Shortcut to avoid having huge lines
	Mat_<int16_t>& gradient = this->adjustmentRingGradient;
	Mat_<float>& snake = this->adjustmentSnake;

	// Find the points where the vertical gradient is maximum
	for (int x = 0; x < gradient.cols; x++) {
		int maxGrad = INT_MIN;
		int bestY = 0;
		for (int y = 0; y < gradient.rows; y++) {
			int gxy = gradient(y, x);
			if (gxy > maxGrad) {
				maxGrad = gxy;
				bestY = y;
			}

			/*// A maximum in the gradient may have been caused by the reflections
			// of the infrared LEDs. In this case, default to the original circle
			bool hasInfraredLed = false;
			hasInfraredLed = hasInfraredLed || (this->adjustmentRing(bestY, x) > infraredThreshold);
			hasInfraredLed = hasInfraredLed || (bestY-1 >= 0 && this->adjustmentRing(bestY-1, x) > infraredThreshold);
			hasInfraredLed = hasInfraredLed || (bestY+1 < this->adjustmentRing.rows && this->adjustmentRing(bestY+1, x) > infraredThreshold);
			if (hasInfraredLed) {
				// The middle point is where the original circular contour passes through
				bestY = gradient.rows/2;
			}*/
		}

		snake(0, x) = bestY;
	}

	if (hasInfrarred) {
		int m0=0, m1=0, s0=0, s1=0;
		//int xinf = 0, xsup = snake.cols;
		int xinf = x0-3, xsup = x1+4;

		for (int x = xinf; x < x0; x++) {
			m0 += snake(0, x);
			s0++;
		}
		for (int x = x1+1; x < xsup; x++) {
			m1 += snake(0, x);
			s1++;
		}

		if (s0 == 0) s0 = 1;
		if (s1 == 0) s1 = 1;

		m0 = m0/s0;
		m1 = m1/s1;

		for (int x = x0; x <= x1; x++) {
			snake(0, x) = m0 + (x-x0)*(m1-m0)/(x1-x0);			// Linear interpolation
		}
	}

	// Smooth the snake
	tools::smoothSnakeFourier(snake, 5);

	/*Tools::smoothSnakeFourier(snake, 3);
	int delta = gradient.rows * 0.1;

	// Improve the estimation
	for (int x = 0; x < gradient.cols; x++) {
		int maxGrad = INT_MIN;
		int bestY = 0;
		int v = snake(0, x);
		int ymin = max(0, v - delta);
		int ymax = min(gradient.rows, v + delta);
		for (int y = ymin; y < ymax; y++) {
			int gxy = gradient(y,x);
			if (gxy > maxGrad) {
				maxGrad = gxy;
				bestY = y;
			}
		}

		snake(0, x) = bestY;
	}

	Tools::smoothSnakeFourier(snake, 5);*/

	// Use the snake to calculate the quality of the pupil border
	this->pupilContourQuality = this->calculatePupilContourQuality(this->adjustmentRing, this->adjustmentRingGradient, snake);

	// Now, transform the points from the ring coordinates to the image coordinates
	Contour result(snake.cols);
	for (int x = 0; x < gradient.cols; x++) {
		int y = snake(0, x);
		double theta = (double(x) / double(snake.cols)) * 2.0 * M_PI;
		double radius = (double(y) / double(gradient.rows - 1))
				* double(radiusMax - radiusMin) + double(radiusMin);
		radius += 1;

		int ximag = int(double(approximateCircle.center.x) + cos(theta) * radius);
		int yimag = int(double(approximateCircle.center.y) + sin(theta) * radius);

		result[x] = Point(ximag, yimag);
	}

	return result;
}

Circle PupilSegmentator::cascadedIntegroDifferentialOperator(const GrayscaleImage& image, Rect ROI)
{
	int minradabs = this->parameters.minimumPupilRadius;
	int minrad = minradabs;
	int maxrad = this->parameters.maximumPupilRadius;

	bool useROI = (ROI.width > 0);
	int minx, miny, maxx, maxy;

	if (useROI) {
		minx = ROI.x;
		maxx = ROI.x+ROI.width;
		miny = ROI.y;
		maxy = ROI.y+ROI.height;
	} else {
		// Exclude the image borders
        int dx = image.cols/10;
        int dy = image.rows/10;

		minx = dx;
		maxx = image.cols-dx;
		miny = dy;
		maxy = image.rows-dy;
	}

	int x, y;
	//int maxStep = INT_MIN;
	int bestX = 0, bestY = 0, bestRadius = 0;

	vector<int> steps(3), radiusSteps(3);
	steps[0] = 10;
	steps[1] = 3;
	steps[2] = 1;
	radiusSteps[0] = 15;
	radiusSteps[1] = 3;
	radiusSteps[2] = 1;

	for (size_t i = 0; i < steps.size(); i++) {
		int maxStep = INT_MIN;
		for (x = minx; x < maxx; x += steps[i]) {
			for (y = miny; y < maxy; y += steps[i]) {
				MaxAvgRadiusResult res = this->maxAvgRadius(image, x, y, minrad, maxrad, radiusSteps[i]);
				if (res.maxStep > maxStep) {
					maxStep = res.maxStep;
					bestX = x;
					bestY = y;
					bestRadius = res.maxRad;
				}
			}
		}

		minx = max<int>(bestX - steps[i], 0);
		maxx = min<int>(bestX + steps[i], image.cols);
		miny = max<int>(bestY - steps[i], 0);
		maxy = min<int>(bestY + steps[i], image.rows);
		minrad = max<int>(bestRadius - radiusSteps[i], minradabs);
		maxrad = bestRadius + radiusSteps[i];

	}

	Circle bestCircle;
	bestCircle.center.x = bestX;
	bestCircle.center.y = bestY;
	bestCircle.radius = bestRadius;

	return bestCircle;
}

PupilSegmentator::MaxAvgRadiusResult PupilSegmentator::maxAvgRadius(const GrayscaleImage& image, int x, int y, int radmin, int radmax, int radstep)
{
	int maxDifference, difference;
	uint8_t actualAvg, nextAvg;
	MaxAvgRadiusResult result;

	maxDifference = INT_MIN;

	actualAvg = this->circleAverage(image, x, y, radmin);
	for (int radius = radmin; radius <= radmax - radstep; radius += radstep) {
		nextAvg = this->circleAverage(image, x, y, radius + radstep);
		difference = int(actualAvg) - int(nextAvg);
		if (difference > maxDifference) {
			maxDifference = difference;
			result.maxRad = radius;
		}

		actualAvg = nextAvg;
	}

	result.maxStep = maxDifference;

	return result;
}

uint8_t PupilSegmentator::circleAverage(const GrayscaleImage& image, int xc, int yc, int rc)
{
	// Optimized Bresenham algorithm for circles
	int x = 0;
	int y = rc;
	int d = 3 - 2 * rc;
	int i, w;
	const uint8_t *row1, *row2, *row3, *row4;
	unsigned S, n;

	i = 0;
	n = 0;
	S = 0;

	int width = image.cols, height = image.rows;

	if ((xc + rc) >= width || (xc - rc) < 0 || (yc + rc)
			>= height || (yc - rc) < 0) {
		while (x < y) {
			i++;
			w = (i - 1) * 8 + 1;

			row1 = image.ptr(yc+y);
			row2 = image.ptr(yc-y);
			row3 = image.ptr(yc+x);
			row4 = image.ptr(yc-x);

			bool row1in = ((yc + y) >= 0 && (yc + y) < height);
			bool row2in = ((yc - y) >= 0 && (yc - y) < height);
			bool row3in = ((yc + x) >= 0 && (yc + x) < height);
			bool row4in = ((yc - x) >= 0 && (yc - y) < height);
			bool xcMxin = ((xc + x) >= 0 && (xc + x) < width);
			bool xcmxin = ((xc - x) >= 0 && (xc - x) < width);
			bool xcMyin = ((xc + y) >= 0 && (xc + y) < width);
			bool xcmyin = ((xc - y) >= 0 && (xc - y) < width);

			if (row1in && xcMxin) {
				S += unsigned(row1[xc + x]);
				n++;
			}
			if (row1in && xcmxin) {
				S += unsigned(row1[xc - x]);
				n++;
			}
			if (row2in && xcMxin) {
				S += unsigned(row2[xc + x]);
				n++;
			}
			if (row2in && xcmxin) {
				S += unsigned(row2[xc - x]);
				n++;
			}
			if (row3in && xcMyin) {
				S += unsigned(row3[xc + y]);
				n++;
			}
			if (row3in && xcmyin) {
				S += unsigned(row3[xc - y]);
				n++;
			}
			if (row4in && xcMyin) {
				S += unsigned(row4[xc + y]);
				n++;
			}
			if (row4in && xcmyin) {
				S += unsigned(row4[xc - y]);
				n++;
			}

			if (d < 0) {
				d = d + (4 * x) + 6;
			} else {
				d = d + 4 * (x - y) + 10;
				y--;
			}

			x++;
		}
	} else {
		while (x < y) {
			i++;
			w = (i - 1) * 8 + 1;

			row1 = image.ptr(yc+y);
			row2 = image.ptr(yc-y);
			row3 = image.ptr(yc+x);
			row4 = image.ptr(yc-x);

			S += unsigned(row1[xc + x]);
			S += unsigned(row1[xc - x]);
			S += unsigned(row2[xc + x]);
			S += unsigned(row2[xc - x]);
			S += unsigned(row3[xc + y]);
			S += unsigned(row3[xc - y]);
			S += unsigned(row4[xc + y]);
			S += unsigned(row4[xc - y]);
			n += 8;

			if (d < 0) {
				d = d + (4 * x) + 6;
			} else {
				d = d + 4 * (x - y) + 10;
				y--;
			}

			x++;
		}
	}

	return (uint8_t) (S / n);
}

void PupilSegmentator::similarityTransform()
{
	double sigma = this->parameters.sigmaPupil;
	double mu = this->parameters.muPupil;

	if (this->_lastSigma != sigma || this->_lastMu != mu) {
		// Rebuild the lookup table
		this->_lastSigma = sigma;
		this->_lastMu = mu;

		this->_LUT = Mat(Size(256,1), CV_8U);
		uchar* pLUT = this->_LUT.ptr(0);

		double num, denom = 2.0 * sigma * sigma;
		double res;

		for (int i = 0; i < 256; i++) {
			num = (double(i) - mu) * (double(i) - mu);
			res = exp(-num / denom) * 255.0;
			pLUT[i] = (uchar) (res);
		}
	}

	LUT(this->equalizedImage, this->_LUT, this->similarityImage);
}

int PupilSegmentator::calculatePupilContourQuality(const GrayscaleImage& region, const Mat_<uint16_t>& regionGradient, const Mat_<float>& contourSnake)
{
	assert(regionGradient.cols == contourSnake.cols);
	assert(region.size() == regionGradient.size());

	int infraredThreshold = this->parameters.infraredThreshold;

	int delta = region.rows * 0.1;
	//const int delta = 2;

	double sum2 = 0;
	double norm2 = 0;
	double v;
	for (int x = 0; x < regionGradient.cols; x++) {
		// Skip this row if there's an infrared reflection
		bool skip = false;
		for (int y = 0; y < region.rows; y++) {
			if (region(y,x) >= infraredThreshold) {
				skip = true;
				break;
			}
		}
		if (skip) continue;

		int yborder = int(contourSnake(0, x));
		int ymin = max(0, yborder-delta);
		int ymax = min(regionGradient.rows, yborder+delta);

		if (yborder < 0) return 0;

		for (int y = 0; y < regionGradient.rows; y++) {
			v = regionGradient(y,x);
			norm2 += v*v;
			if (y >= ymin && y < ymax) {
				sum2 += v*v;
			}
		}
	}

	if (norm2 == 0.0) {
		return 0;
	}

	assert(sum2 <= norm2);
	assert(norm2 > 0);

	return int((100.0*sum2)/norm2);
}
