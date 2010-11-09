/*
 * File:   pupilsegmentator.cpp
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:39 PM
 */

#include "pupilsegmentator.h"
#include "tools.h"
#include "tools.h"

PupilSegmentator::PupilSegmentator()
{
	this->_lastSigma = this->_lastMu = -100.0;
}

PupilSegmentator::~PupilSegmentator()
{
}

ContourAndCloseCircle PupilSegmentator::segmentPupil(const Mat& image)
{
	assert(image.channels() == 1 && image.depth() == CV_8U);

	this->setupBuffers(image);
	ContourAndCloseCircle result;

	Circle pupilCircle = this->approximatePupil(this->workingImage);

	pupilCircle.radius /= this->resizeFactor;
	pupilCircle.xc /= this->resizeFactor;
	pupilCircle.yc /= this->resizeFactor;

	result.first = this->adjustPupilContour(image, pupilCircle);
	result.second = Tools::approximateCircle(result.first);

	return result;

}

void PupilSegmentator::setupBuffers(const Mat& image)
{
	Parameters* parameters = Parameters::getParameters();

	// Initialize the working image
	int bufferWidth = parameters->bufferWidth;
	int width = image.cols, height = image.rows;

	this->resizeFactor = ((width > bufferWidth) ? double(bufferWidth) / double(width) : 1.0);

	if (this->resizeFactor != 1.0) {
		resize(image, this->workingImage, Size(), this->resizeFactor, this->resizeFactor);
	} else {
		this->workingImage = image;
	}

	if (!this->ROI.width || !this->ROI.height) {
		this->workingROI = Rect(0, 0, this->workingImage.cols, this->workingImage.rows);
	} else {
		this->workingROI = Rect(this->ROI.x*this->resizeFactor, this->ROI.y*this->resizeFactor, this->ROI.width*this->resizeFactor, this->ROI.height*this->resizeFactor);
	}


	this->adjustmentRing.create(Size(parameters->pupilAdjustmentRingWidth, parameters->pupilAdjustmentRingHeight));
	this->adjustmentRingGradient.create(Size(parameters->pupilAdjustmentRingWidth, parameters->pupilAdjustmentRingHeight));
	this->adjustmentSnake.create(Size(parameters->pupilAdjustmentRingWidth, 1));
}

Circle PupilSegmentator::approximatePupil(const Mat_<uint8_t>& image)
{
	// First, equalize the image
	equalizeHist(image, this->equalizedImage);

	// Then apply the similarity transformation
	this->similarityTransform();
	blur(this->similarityImage, this->similarityImage, Size(3, 3));

	// Now perform the cascaded integro-differential operator
	Circle res = this->cascadedIntegroDifferentialOperator(this->similarityImage);
	return res;
}

Contour PupilSegmentator::adjustPupilContour(const Mat_<uint8_t>& image, const Circle& approximateCircle)
{
	int radiusMin = approximateCircle.radius * 0.5, radiusMax =
			approximateCircle.radius * 1.5;
	Tools::extractRing(image, this->adjustmentRing,
			approximateCircle.xc, approximateCircle.yc, radiusMin, radiusMax);

	int infraredThreshold = Parameters::getParameters()->infraredThreshold;


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

	if (x0 > INT_MIN && x1 < INT_MAX) {
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

	if (x0 > INT_MIN && x1 < INT_MAX) {
		for (int x = x0; x <= x1; x++) {
			snake(0, x) = snake(0, x0-1) + (x-x0)*((snake(0, x1+1)-snake(0, x0-1)) / (x1-x0));
		}
	}

	// Smooth the snake
	Tools::smoothSnakeFourier(snake, 5);

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

		int ximag = int(double(approximateCircle.xc) + cos(theta) * radius);
		int yimag = int(double(approximateCircle.yc) + sin(theta) * radius);

		result[x] = Point(ximag, yimag);
	}

	return result;
}

Circle PupilSegmentator::cascadedIntegroDifferentialOperator(const Mat_<uint8_t>& image)
{
	const Parameters* parameters = Parameters::getParameters();

	int minradabs = parameters->minimumPupilRadius;
	int minrad = minradabs;
	int maxrad = parameters->maximumPupilRadius;

	int dx = image.cols*0.2, dy = image.rows*0.2;			// Exclude the image borders

	/*int minx = dx, miny = dy;
	int maxx = image.cols - dx, maxy = image.rows - dy;*/
	int minx = this->workingROI.x+dx, miny = this->workingROI.y+dy;			// Detect the circle ONLY inside the ROI
	int maxx = this->workingROI.x+this->workingROI.width-dx, maxy = this->workingROI.y+this->workingROI.height-dy;
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
				MaxAvgRadiusResult res = this->maxAvgRadius(image, x, y,
						minrad, maxrad, radiusSteps[i]);
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
	bestCircle.xc = bestX;
	bestCircle.yc = bestY;
	bestCircle.radius = bestRadius;

	return bestCircle;
}

PupilSegmentator::MaxAvgRadiusResult PupilSegmentator::maxAvgRadius(const Mat_<uint8_t>& image, int x, int y, int radmin, int radmax, int radstep)
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

uint8_t PupilSegmentator::circleAverage(const Mat_<uint8_t>& image, int xc, int yc, int rc)
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

			/*row1 = ((uint8_t*) (image->imageData)) + image->widthStep
					* (yc + y);
			row2 = ((uint8_t*) (image->imageData)) + image->widthStep
					* (yc - y);
			row3 = ((uint8_t*) (image->imageData)) + image->widthStep
					* (yc + x);
			row4 = ((uint8_t*) (image->imageData)) + image->widthStep
					* (yc - x);*/
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

			/*row1 = ((uint8_t*) (image->imageData)) + image->widthStep
					* (yc + y);
			row2 = ((uint8_t*) (image->imageData)) + image->widthStep
					* (yc - y);
			row3 = ((uint8_t*) (image->imageData)) + image->widthStep
					* (yc + x);
			row4 = ((uint8_t*) (image->imageData)) + image->widthStep
					* (yc - x);
					*/

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
	Parameters* parameters = Parameters::getParameters();

	double sigma = parameters->sigmaPupil;
	double mu = parameters->muPupil;

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

int PupilSegmentator::calculatePupilContourQuality(const Mat_<uint8_t>& region, const Mat_<uint16_t>& regionGradient, const Mat_<float>& contourSnake)
{
	assert(regionGradient.cols == contourSnake.cols);
	assert(region.size() == regionGradient.size());

	int infraredThreshold = Parameters::getParameters()->infraredThreshold;

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

	if (!norm2) {
		return 0;
	}

	assert(sum2 < norm2);
	assert(norm2 > 0);

	return int((100.0*sum2)/norm2);
}
