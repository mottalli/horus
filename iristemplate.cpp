/*
 * iristemplate.cpp
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#include "iristemplate.h"
#include <iostream>

inline unsigned char setBit(unsigned char b, int bit, bool value);
inline bool getBit(unsigned char b, int bit);

IrisTemplate::IrisTemplate()
{
	this->irisTemplate = NULL;
	this->mask = NULL;
}

IrisTemplate::IrisTemplate(const CvMat* binaryTemplate, const CvMat* binaryMask)
{
	assert(binaryTemplate->width % 8 == 0);
	assert(binaryMask->width% 8 == 0);

	this->irisTemplate = cvCreateMat(binaryTemplate->height, binaryTemplate->width/8, CV_8U);
	this->mask = cvCreateMat(binaryMask->height, binaryMask->width/8, CV_8U);

	this->packBits(binaryTemplate, this->irisTemplate);
	this->packBits(binaryMask, this->mask);
}

IrisTemplate::IrisTemplate(const IrisTemplate& otherTemplate)
{
	this->irisTemplate = (CvMat*)cvClone(otherTemplate.irisTemplate);
	this->mask = (CvMat*)cvClone(otherTemplate.mask);
}

IrisTemplate& IrisTemplate::operator=(const IrisTemplate& otherTemplate)
{
	if (this->irisTemplate != NULL) {
		cvReleaseMat(&this->irisTemplate);
	}
	if (this->mask != NULL) {
		cvReleaseMat(&this->mask);
	}

	this->irisTemplate = (CvMat*)cvClone(otherTemplate.irisTemplate);
	this->mask = (CvMat*)cvClone(otherTemplate.mask);

	return *this;
}

IrisTemplate::~IrisTemplate()
{
	if (this->irisTemplate != NULL) {
		cvReleaseMat(&this->irisTemplate);
		cvReleaseMat(&this->mask);
	}
}

Image* IrisTemplate::getTemplate() const
{
	CvMat* foo = cvCreateMat(this->irisTemplate->height, this->irisTemplate->width*8, CV_8U);
	Image* img = new Image;
	this->unpackBits(this->irisTemplate, foo, 255);
	return cvGetImage(foo, img);
}

Image* IrisTemplate::getNoiseMask() const
{
	CvMat* foo = cvCreateMat(this->mask->height, this->irisTemplate->width*8, CV_8U);
	Image* img = new Image;
	this->unpackBits(this->mask, foo, 255);
	return cvGetImage(foo, img);
}

// Pack the binary in src into bits
void IrisTemplate::packBits(const CvMat* src, CvMat* dest) const
{
	assert(src->width / 8 == dest->width);
	assert(src->height == dest->height);

	for (int y = 0; y < src->height; y++) {
		const unsigned char* srcrow = &(src->data.ptr[y*src->step]);

		int xsrc = 0;
		for (int bytenum = 0; bytenum < dest->width; bytenum++) {
			unsigned char *destbyte =  &(dest->data.ptr[y*dest->step+bytenum]);
			unsigned char byteval = 0;
			for (int bit = 0; bit < 8; bit++) {
				bool value = (srcrow[xsrc] > 0 ? true : false);
				byteval = setBit(byteval, bit, value);
				xsrc++;
			}
			*destbyte = byteval;
		}
	}
}

void IrisTemplate::unpackBits(const CvMat* src, CvMat* dest, int trueval) const
{
	assert(src->width * 8 == dest->width);
	assert(src->height == dest->height);

	for (int y = 0; y < src->height; y++) {
		int xdest = 0;
		for (int xsrc = 0; xsrc < src->width; xsrc++) {
			unsigned char byte = src->data.ptr[y*src->step+xsrc];
			for (int bit = 0; bit < 8; bit++) {
				cvSetReal2D(dest, y, xdest, getBit(byte, bit) ? trueval : 0);
				xdest++;
			}
		}
	}
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

static unsigned char BIT_MASK[] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01 };

unsigned char setBit(unsigned char b, int bit, bool value)
{
	if (value) {
		// Set to 1
		return b | BIT_MASK[bit];
	} else {
		// Set to 0
		return b & (~BIT_MASK[bit]);
	}
}

bool getBit(unsigned char b, int bit)
{
	return (b & BIT_MASK[bit]) ? true : false;
}
