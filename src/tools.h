/*
 * tools.h
 *
 *  Created on: Jun 15, 2009
 *      Author: marcelo
 */

#pragma once

#include "common.h"
#include "external/base64.h"
#include "segmentator.h"
#include "iristemplate.h"
#include "irisencoder.h"

namespace horus {

namespace tools {

void packBits(const GrayscaleImage& src, GrayscaleImage& dest);
void unpackBits(const GrayscaleImage& src, GrayscaleImage& dest, int trueval = 1);

void extractRing(const GrayscaleImage& src, GrayscaleImage& dest, int x0, int y0, int radiusMin, int radiusMax);
void smoothSnakeFourier(Mat_<float>& snake, int coefficients);
Circle approximateCircle(const Contour& contour);

// Useful debugging functions
void drawHistogram(const IplImage* img);

template<class T>
std::string base64EncodeMat(const Mat& mat);
template<class T>
Mat_<T> base64DecodeMat(const std::string &s);

void stretchHistogram(const Image& image, Image& dest, float marginMin=0.0, float marginMax=0.0);

std::vector< std::pair<Point, Point> > iterateIris(const SegmentationResult& segmentation, int width, int height, double theta0=0.0, double theta1=2.0*M_PI, double radiusMin = 0.0, double radiusMax=1.0);
void superimposeTexture(GrayscaleImage& image, const GrayscaleImage& texture, const SegmentationResult& segmentation, double theta0=0.0, double theta1=2.0*M_PI, double radius=1.0, bool blend=true, double blendStart = 0.7);

GrayscaleImage normalizeImage(const GrayscaleImage& image, uint8_t min=0, uint8_t max=255);	// Normalizes an image to the given range
void toGrayscale(const Image& src, GrayscaleImage& dest, bool cloneIfAlreadyGray);

/*
 10000000: 128
 01000000: 64
 00100000: 32
 00010000: 16
 00001000: 8
 00000100: 4
 00000010: 2
 00000001: 1
 */

const uint8_t BIT_MASK[] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01 };
inline uint8_t setBit(uint8_t b, int bit, bool value);
inline bool getBit(uint8_t b, int bit);

}
}



template<class T>
string horus::tools::base64EncodeMat(const Mat& mat)
{
	int width = mat.cols, height = mat.rows;
	assert(mat.channels() == 1);
	assert(mat.isContinuous());

	unsigned char* buffer = new unsigned char[2*sizeof(int16_t) + width*height*sizeof(T)];		// Stores width, height and data

	// Store width and height
	int16_t* header = (int16_t*)buffer;
	header[0] = width;
	header[1] = height;

	T* p = (T*)(buffer + 2*sizeof(int16_t));		// Pointer to the actual data past the width and height

	for (int y = 0; y < height; y++) {
		memcpy((void*)(p + y*width), mat.ptr(y), width*sizeof(T));		// Copy one line
	}

	string base64 = Tools::base64Encode(buffer, 2*sizeof(int16_t) + width*height*sizeof(T));

	delete[] buffer;

	return base64;
}

template<class T>
Mat_<T> horus::tools::base64DecodeMat(const string &s)
{
	int width, height;

	string decoded = Tools::base64Decode(s);
	const char* buffer = decoded.c_str();

	int16_t* header = (int16_t*)buffer;
	width = header[0];
	height = header[1];

	T* p = (T*)(buffer + 2*sizeof(int16_t));		// Pointer to the actual data past the width and height

	Mat_<T> res(height, width);

	for (int y = 0; y < height; y++) {
		memcpy(res.ptr(y), p + y*width, width*sizeof(T));		// Copy one line
	}

	return res;
}
