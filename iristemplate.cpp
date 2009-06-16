/*
 * iristemplate.cpp
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#include <iostream>
#include "iristemplate.h"
#include "tools.h"

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

	Tools::packBits(binaryTemplate, this->irisTemplate);
	Tools::packBits(binaryMask, this->mask);
}

IrisTemplate::IrisTemplate(const IrisTemplate& otherTemplate)
{
	this->irisTemplate = cvCloneMat(otherTemplate.irisTemplate);
	this->mask = cvCloneMat(otherTemplate.mask);
}

IrisTemplate& IrisTemplate::operator=(const IrisTemplate& otherTemplate)
{
	if (this->irisTemplate != NULL) {
		cvReleaseMat(&this->irisTemplate);
		cvReleaseMat(&this->mask);
	}

	this->irisTemplate = cvCloneMat(otherTemplate.irisTemplate);
	this->mask = cvCloneMat(otherTemplate.mask);

	return *this;
}

IrisTemplate::~IrisTemplate()
{
	if (this->irisTemplate != NULL) {
		cvReleaseMat(&this->irisTemplate);
		cvReleaseMat(&this->mask);
	}
}

Image* IrisTemplate::getTemplateImage() const
{
	CvMat* foo = cvCreateMat(this->irisTemplate->height, this->irisTemplate->width*8, CV_8U);
	Tools::unpackBits(this->irisTemplate, foo, 255);
	Image* img = cvCreateImage(cvGetSize(foo), IPL_DEPTH_8U, 1);
	cvCopy(foo, img);

	cvReleaseMat(&foo);
	return img;
}

Image* IrisTemplate::getNoiseMaskImage() const
{
	CvMat* foo = cvCreateMat(this->mask->height, this->mask->width*8, CV_8U);
	Tools::unpackBits(this->mask, foo, 255);
	Image* img = cvCreateImage(cvGetSize(foo), IPL_DEPTH_8U, 1);
	cvCopy(foo, img);

	cvReleaseMat(&foo);
	return img;
}

CvMat* IrisTemplate::getUnpackedTemplate() const
{
	CvMat* unpacked = cvCreateMat(this->irisTemplate->height, this->irisTemplate->width*8, CV_8U);
	Tools::unpackBits(this->irisTemplate, unpacked);
	return unpacked;
}

CvMat* IrisTemplate::getUnpackedMask() const
{
	CvMat* unpacked = cvCreateMat(this->mask->height, this->mask->width*8, CV_8U);
	Tools::unpackBits(this->mask, unpacked);
	return unpacked;
}
