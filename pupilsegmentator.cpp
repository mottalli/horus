/*
 * File:   pupilsegmentator.cpp
 * Author: marcelo
 *
 * Created on January 21, 2009, 8:39 PM
 */

#include "pupilsegmentator.h"
#include "helperfunctions.h"
#include <cmath>

#define SAME_SIZE(im1, im2) (im1->width==im2->width && im1->height == im2->height)

PupilSegmentator::PupilSegmentator() {
    this->buffers.LUT = cvCreateMat(256, 1, CV_8UC1);
    this->buffers.similarityImage = NULL;
    this->_lastSigma = this->_lastMu = -100.0;

    this->buffers.workingImage = NULL;

    this->buffers.adjustmentRing = NULL;
    this->buffers.adjustmentRingGradient = NULL;
}

PupilSegmentator::~PupilSegmentator() {
    cvReleaseMat(&this->buffers.LUT);

    if (this->buffers.similarityImage != NULL) {
        cvReleaseImage(&this->buffers.similarityImage);
    }
}

ContourAndCloseCircle PupilSegmentator::segmentPupil(const Image* image) {
    this->setupBuffers(image);
    ContourAndCloseCircle result;

    Circle pupilCircle = this->approximatePupil(this->buffers.workingImage);

    pupilCircle.radius /= this->buffers.resizeFactor;
    pupilCircle.xc /= this->buffers.resizeFactor;
    pupilCircle.yc /= this->buffers.resizeFactor;

    result.first = this->adjustPupilContour(image, pupilCircle);
    result.second = HelperFunctions::approximateCircle(result.first);

    return result;

}

void PupilSegmentator::setupBuffers(const Image* image) {
	Parameters* parameters = Parameters::getParameters();

    // Initialize the working image
    int bufferWidth = parameters->bufferWidth;

    int workingWidth, workingHeight;
    int width = image->width, height = image->height;
    double resizeFactor;

    if (image->width > bufferWidth) {
        resizeFactor = double(bufferWidth) / double(image->width);
        workingWidth = int(double(width)*resizeFactor);
        workingHeight = int(double(height)*resizeFactor);
    } else {
        resizeFactor = 1.0;
        workingWidth = width;
        workingHeight = height;
    }

    this->buffers.resizeFactor = resizeFactor;

    Image*& workingImage = this->buffers.workingImage;

    if (workingImage == NULL || workingImage->width != workingWidth || workingImage->height != workingHeight) {
        if (workingImage != NULL) {
            cvReleaseImage(&workingImage);
        }

        workingImage = cvCreateImage(cvSize(workingWidth, workingHeight), IPL_DEPTH_8U, 1);
    }


    if (resizeFactor == 1.0) {
		cvCopy(image, workingImage);
	} else {
		cvResize(image, workingImage, CV_INTER_LINEAR);
	}



    if (this->buffers.similarityImage == NULL || !SAME_SIZE(this->buffers.similarityImage, workingImage)) {
    	if (this->buffers.similarityImage != NULL) {
    		cvReleaseImage(&this->buffers.similarityImage);
    		cvReleaseImage(&this->buffers.equalizedImage);
    	}
		this->buffers.similarityImage = cvCreateImage(cvGetSize(workingImage), IPL_DEPTH_8U, 1);
		this->buffers.equalizedImage = cvCreateImage(cvGetSize(workingImage), IPL_DEPTH_8U, 1);
    }

    if (this->buffers.adjustmentRing == NULL || this->buffers.adjustmentRing->width != parameters->pupilAdjustmentRingWidth || this->buffers.adjustmentRing->height != parameters->pupilAdjustmentRingHeight) {
    	if (this->buffers.adjustmentRing != NULL) {
    		cvReleaseImage(&this->buffers.adjustmentRing);
    		cvReleaseImage(&this->buffers.adjustmentRingGradient);
    		cvReleaseMat(&this->buffers.adjustmentSnake);
    	}

    	this->buffers.adjustmentRing = cvCreateImage(cvSize(parameters->pupilAdjustmentRingWidth, parameters->pupilAdjustmentRingHeight), IPL_DEPTH_16S, 1);
    	this->buffers.adjustmentRingGradient = cvCreateImage(cvSize(parameters->pupilAdjustmentRingWidth, parameters->pupilAdjustmentRingHeight), IPL_DEPTH_16S, 1);
    	this->buffers.adjustmentSnake = cvCreateMat(1, parameters->pupilAdjustmentRingWidth, CV_32F);
    }

}

Circle PupilSegmentator::approximatePupil(const Image* image) {
    // First, equalize the image
    cvEqualizeHist(image, this->buffers.equalizedImage);

    // Then apply the similarity transformation
    this->similarityTransform();
    cvSmooth(this->buffers.similarityImage, this->buffers.similarityImage, CV_GAUSSIAN, 13);

    // Now perform the cascaded integro-differential operator
    return this->cascadedIntegroDifferentialOperator(this->buffers.similarityImage);

}

Contour PupilSegmentator::adjustPupilContour(const Image* image, const Circle& approximateCircle)
{
	int radiusMin = approximateCircle.radius*0.5, radiusMax = approximateCircle.radius*1.5;
	HelperFunctions::extractRing(image, this->buffers.adjustmentRing, approximateCircle.xc, approximateCircle.yc, radiusMin, radiusMax);

	// Calculate the vertical gradient
	cvSobel(this->buffers.adjustmentRing, this->buffers.adjustmentRingGradient, 0, 1, 3);
	cvSmooth(this->buffers.adjustmentRingGradient, this->buffers.adjustmentRingGradient, CV_GAUSSIAN, 3, 3);

	// Shortcut to avoid having huge lines
	Image* gradient = this->buffers.adjustmentRingGradient;
	CvMat* snake = this->buffers.adjustmentSnake;

	// Find the points where the vertical gradient is maximum
	for (int x = 0; x < gradient->width; x++) {
		int maxGrad = INT_MIN;
		int bestY = 0;
		for (int y = 0; y < gradient->height; y++) {
			int gxy = cvGetReal2D(gradient, y, x);
			if  (gxy > maxGrad) {
				maxGrad = gxy;
				bestY = y;
			}
		}

		cvSetReal2D(snake, 0, x, bestY);
	}

	// Smooth the snake
	HelperFunctions::smoothSnakeFourier(snake, 3);

	// Improve the estimation
	for (int x = 0; x < gradient->width; x++) {
		int maxGrad = INT_MIN;
		int bestY = 0;
		int v = cvGetReal2D(snake, 0, x);
		int ymin = std::max(0, v-5);
		int ymax = std::min(gradient->height, v+5);
		for (int y = ymin; y < ymax; y++) {
			int gxy = cvGetReal2D(gradient, y, x);
			if (gxy > maxGrad) {
				maxGrad = gxy;
				bestY = y;
			}
		}

		cvSetReal2D(snake, 0, x, bestY);
	}

	HelperFunctions::smoothSnakeFourier(snake, 5);

	// Now, transform the points from the ring coordinates to the image coordinates
	Contour result(snake->cols);
	for (int x = 0; x < gradient->width; x++) {
		int y = cvGetReal2D(snake, 0, x);
		double theta = (double(x)/double(snake->cols))*2.0*M_PI;
		double radius = (double(y)/double(gradient->height-1))*double(radiusMax-radiusMin) + double(radiusMin);

		int ximag = int(double(approximateCircle.xc) + std::cos(theta)*radius);
		int yimag = int(double(approximateCircle.yc) + std::sin(theta)*radius);

		result[x] = cvPoint(ximag,yimag);
	}

	return result;
}

Circle PupilSegmentator::cascadedIntegroDifferentialOperator(const Image* image) {
    int minrad = 10, minradabs = 10;
    int maxrad = 80;
    int minx = 10, miny = 10;
    int maxx = image->width-10, maxy = image->height-10;
    int x, y, radius;
    //int maxStep = INT_MIN;
    int bestX, bestY, bestRadius;

    std::vector<int> steps(3), radiusSteps(3);
    steps[0] = 10; steps[1] = 3; steps[2] = 1;
    radiusSteps[0] = 15; radiusSteps[1] = 3; radiusSteps[2] = 1;

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

        minx = std::max<int>(bestX-steps[i], 0);
        maxx = std::min<int>(bestX+steps[i], image->width);
        miny = std::max<int>(bestY-steps[i], 0);
        maxy = std::min<int>(bestY+steps[i], image->height);
        minrad = std::max<int>(bestRadius-radiusSteps[i], minradabs);
        maxrad = bestRadius+radiusSteps[i];

    }

    Circle bestCircle;
    bestCircle.xc = bestX;
    bestCircle.yc = bestY;
    bestCircle.radius = bestRadius;

    return bestCircle;
}

PupilSegmentator::MaxAvgRadiusResult PupilSegmentator::maxAvgRadius(const Image* image, int x, int y, int radmin, int radmax, int radstep)
{
    int maxDifference, difference;
    unsigned char actualAvg, nextAvg;
    MaxAvgRadiusResult result;

    maxDifference = INT_MIN;

    actualAvg = this->circleAverage(image, x, y, radmin);
    for (int radius = radmin; radius <= radmax-radstep; radius += radstep) {
        nextAvg = this->circleAverage(image, x, y, radius+radstep);
        difference = int(actualAvg)-int(nextAvg);
        if (difference > maxDifference) {
            maxDifference = difference;
            result.maxRad = radius;
        }

        actualAvg = nextAvg;
    }

    result.maxStep = maxDifference;

    return result;
}

unsigned char PupilSegmentator::circleAverage(const Image* image, int xc, int yc, int rc) {
    // Optimized Bresenham algorithm for circles
    int x = 0;
    int y = rc;
    int d = 3 - 2 * rc;
    int i, w;
    unsigned char *row1, *row2, *row3, *row4;
    unsigned S, n;

    i = 0;
    n = 0;
    S = 0;

    if ((xc + rc) >= image->width || (xc - rc) < 0 || (yc + rc) >= image->height || (yc - rc) < 0) {
        while (x < y) {
            i++;
            w = (i - 1)*8 + 1;

            row1 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc + y);
            row2 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc - y);
            row3 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc + x);
            row4 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc - x);

            bool row1in = ((yc + y) >= 0 && (yc + y) < image->height);
            bool row2in = ((yc - y) >= 0 && (yc - y) < image->height);
            bool row3in = ((yc + x) >= 0 && (yc + x) < image->height);
            bool row4in = ((yc - x) >= 0 && (yc - y) < image->height);
            bool xcMxin = ((xc + x) >= 0 && (xc + x) < image->width);
            bool xcmxin = ((xc - x) >= 0 && (xc - x) < image->width);
            bool xcMyin = ((xc + y) >= 0 && (xc + y) < image->width);
            bool xcmyin = ((xc - y) >= 0 && (xc - y) < image->width);

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
            w = (i - 1)*8 + 1;

            row1 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc + y);
            row2 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc - y);
            row3 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc + x);
            row4 = ((unsigned char*) (image->imageData)) + image->widthStep * (yc - x);

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

    return (unsigned char) (S / n);
}

void PupilSegmentator::similarityTransform() {
    Parameters* parameters = Parameters::getParameters();

    double sigma = parameters->sigmaPupil;
    double mu = parameters->muPupil;
    double num, denom = 2.0*sigma*sigma;
    double res;

    if (this->_lastSigma != sigma || this->_lastMu != mu) {
        // Rebuild the lookup table
        this->_lastSigma = sigma;
        this->_lastMu = mu;

        unsigned char* pLUT = this->buffers.LUT->data.ptr;
        for (int i = 0; i < 256; i++) {
            num = (double(i)-mu)*(double(i)-mu);
            res = std::exp(-num/denom)*255.0;
            pLUT[i] = (unsigned char)(res);
        }
    }

    cvLUT(this->buffers.equalizedImage, this->buffers.similarityImage, this->buffers.LUT);
}
