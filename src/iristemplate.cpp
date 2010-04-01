/*
 * iristemplate.cpp
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#include <iostream>
#include <sstream>
#include "iristemplate.h"
#include "tools.h"

IrisTemplate::IrisTemplate()
{
	this->irisTemplate = NULL;
	this->mask = NULL;
}

IrisTemplate::IrisTemplate(const Mat& binaryTemplate, const Mat& binaryMask)
{
	assert(binaryTemplate.size() == binaryMask.size());
	assert(binaryTemplate.depth() == CV_8U);
	assert(binaryMask.depth() == CV_8U);
	assert(binaryTemplate.channels() == 1);
	assert(binaryMask.channels() == 1);

	assert(binaryTemplate.cols % 8 == 0);

	Tools::packBits(binaryTemplate, this->irisTemplate);
	Tools::packBits(binaryMask, this->mask);
}

IrisTemplate::IrisTemplate(const IrisTemplate& otherTemplate)
{
	this->irisTemplate = otherTemplate.irisTemplate.clone();
	this->mask = otherTemplate.mask.clone();
}

IrisTemplate& IrisTemplate::operator=(const IrisTemplate& otherTemplate)
{
	this->irisTemplate = otherTemplate.irisTemplate.clone();
	this->mask = otherTemplate.mask.clone();

	return *this;
}

IrisTemplate::~IrisTemplate()
{
}

Mat IrisTemplate::getTemplateImage() const
{
	//CvMat* foo = cvCreateMat(this->irisTemplate->height, this->irisTemplate->width*8, CV_8U);
	Mat_<uint8_t> image;
	Tools::unpackBits(this->irisTemplate, image, 255);
	return Mat(image);
}

Mat IrisTemplate::getNoiseMaskImage() const
{
	Mat_<uint8_t> image;
	Tools::unpackBits(this->mask, image, 255);
	return Mat(image);
}

Mat IrisTemplate::getUnpackedTemplate() const
{
	Mat_<uint8_t> unpacked;
	Tools::unpackBits(this->irisTemplate, unpacked);
	return Mat(unpacked);
}

Mat IrisTemplate::getUnpackedMask() const
{
	Mat_<uint8_t> unpacked;
	Tools::unpackBits(this->mask, unpacked);
	return Mat(unpacked);
}

