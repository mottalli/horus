/*
 * tools.h
 *
 *  Created on: Jun 15, 2009
 *      Author: marcelo
 */

#pragma once

#include "common.h"
#include "segmentationresult.h"

namespace Tools
{
	void packBits(const GrayscaleImage& src, GrayscaleImage& dest);
	void unpackBits(const GrayscaleImage& src, GrayscaleImage& dest, int trueval = 1);

	void extractRing(const GrayscaleImage& src, GrayscaleImage& dest, int x0, int y0, int radiusMin, int radiusMax);
	void smoothSnakeFourier(Mat_<float>& snake, int coefficients);
	Circle approximateCircle(const Contour& contour);

	// Useful debugging functions
	void drawHistogram(const IplImage* img);

	std::string base64EncodeMat(const Mat& mat);
	Mat base64DecodeMat(const std::string &s);

	void stretchHistogram(const GrayscaleImage& image, GrayscaleImage& dest, float marginMin=0.01, float marginMax=0.0);

	std::vector< std::pair<Point, Point> > iterateIris(const SegmentationResult& segmentation, int width, int height, double theta0=0.0, double theta1=2.0*M_PI, double radius=1.0);
	void superimposeTexture(GrayscaleImage& image, const GrayscaleImage& texture, const SegmentationResult& segmentation, double theta0=0.0, double theta1=2.0*M_PI, double radius=1.0, bool blend=true, double blendStart = 0.7);

	GrayscaleImage normalizeImage(const GrayscaleImage& image, uint8_t min=0, uint8_t max=255);	// Normalizes an image to the given range
	void toGrayscale(const Image& src, GrayscaleImage& dest, bool cloneIfAlreadyGray);
}

