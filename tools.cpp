/*
 * tools.cpp
 *
 *  Created on: Jun 15, 2009
 *      Author: marcelo
 */

#include "tools.h"

inline uint8_t setBit(uint8_t b, int bit, bool value);
inline bool getBit(uint8_t b, int bit);

// Pack the binary in src into bits
void Tools::packBits(const CvMat* src, CvMat* dest)
{
	assert(src->width / 8 == dest->width);
	assert(src->height == dest->height);

	for (int y = 0; y < src->height; y++) {
		const uint8_t* srcrow = &(src->data.ptr[y*src->step]);

		int xsrc = 0;
		for (int bytenum = 0; bytenum < dest->width; bytenum++) {
			uint8_t *destbyte =  &(dest->data.ptr[y*dest->step+bytenum]);
			uint8_t byteval = 0;
			for (int bit = 0; bit < 8; bit++) {
				bool value = (srcrow[xsrc] > 0 ? true : false);
				byteval = setBit(byteval, bit, value);
				xsrc++;
			}
			*destbyte = byteval;
		}
	}
}

void Tools::unpackBits(const CvMat* src, CvMat* dest, int trueval)
{
	assert(src->width * 8 == dest->width);
	assert(src->height == dest->height);

	for (int y = 0; y < src->height; y++) {
		int xdest = 0;
		for (int xsrc = 0; xsrc < src->width; xsrc++) {
			uint8_t byte = src->data.ptr[y*src->step+xsrc];
			for (int bit = 0; bit < 8; bit++) {
				cvSetReal2D(dest, y, xdest, getBit(byte, bit) ? trueval : 0);
				xdest++;
			}
		}
	}
}

void Tools::drawHistogram(const IplImage* img)
{
	int bins = 256;
	int hsize[] = { bins };

	float xranges[] = { 0, 256 };
	float* ranges[] =  { xranges };

	IplImage* copy = cvCloneImage(img);
	IplImage* planes[] = { copy };

	CvHistogram* hist = cvCreateHist(1, hsize, CV_HIST_ARRAY, ranges, 1);
	cvCalcHist(planes, hist, false);

	float min_value = 0, max_value = 0;
	cvGetMinMaxHistValue(hist, &min_value, &max_value);

	IplImage* imgHist = cvCreateImage(cvSize(bins, 50), IPL_DEPTH_8U, 1);
	cvSet(imgHist, cvScalar(255,0,0,0));
	for (int i = 0; i < bins; i++) {
		float value = cvQueryHistValue_1D(hist, i);
		int normalized = cvRound(imgHist->height*(value/max_value));
		cvLine(imgHist, cvPoint(i,imgHist->height), cvPoint(i, imgHist->height-normalized), CV_RGB(0,0,0));
	}

	cvNamedWindow("histogram");
	cvShowImage("histogram", imgHist);

	cvReleaseImage(&imgHist);
	cvReleaseHist(&hist);
	cvReleaseImage(&copy);
}

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

static uint8_t BIT_MASK[] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01 };

uint8_t setBit(uint8_t b, int bit, bool value)
{
	if (value) {
		// Set to 1
		return b | BIT_MASK[bit];
	} else {
		// Set to 0
		return b & (~BIT_MASK[bit]);
	}
}

bool getBit(uint8_t b, int bit)
{
	return (b & BIT_MASK[bit]) ? true : false;
}
