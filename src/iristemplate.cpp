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

using namespace horus;

IrisTemplate::IrisTemplate()
{
}

IrisTemplate::IrisTemplate(const GrayscaleImage& binaryTemplate, const GrayscaleImage& binaryMask, string encoderSignature_)
{
	assert(binaryTemplate.size() == binaryMask.size());
	assert(binaryTemplate.depth() == CV_8U);
	assert(binaryMask.depth() == CV_8U);
	assert(binaryTemplate.channels() == 1);
	assert(binaryMask.channels() == 1);

	assert(binaryTemplate.cols % 8 == 0);

	tools::packBits(binaryTemplate, this->irisTemplate);
	tools::packBits(binaryMask, this->mask);

	this->encoderSignature = encoderSignature_;
}

IrisTemplate::IrisTemplate(const IrisTemplate& otherTemplate)
{
	this->irisTemplate = otherTemplate.irisTemplate.clone();
	this->mask = otherTemplate.mask.clone();
	this->encoderSignature = otherTemplate.encoderSignature;
}

IrisTemplate& IrisTemplate::operator=(const IrisTemplate& otherTemplate)
{
	this->irisTemplate = otherTemplate.irisTemplate.clone();
	this->mask = otherTemplate.mask.clone();
	this->encoderSignature = otherTemplate.encoderSignature;

	return *this;
}

IrisTemplate::~IrisTemplate()
{
}

GrayscaleImage IrisTemplate::getTemplateImage() const
{
	GrayscaleImage imgTemplate, imgMask;
	tools::unpackBits(this->irisTemplate, imgTemplate, 255);
	tools::unpackBits(this->mask, imgMask, 255);

	bitwise_not(imgMask, imgMask);			// Hacky way to NOT the template
	imgTemplate.setTo(127, imgMask);

	return imgTemplate;
}

GrayscaleImage IrisTemplate::getUnpackedTemplate() const
{
	GrayscaleImage unpacked;
	tools::unpackBits(this->irisTemplate, unpacked);
	return unpacked;
}

GrayscaleImage IrisTemplate::getUnpackedMask() const
{
	GrayscaleImage unpacked;
	tools::unpackBits(this->mask, unpacked);
	return unpacked;
}

void IrisTemplate::setPackedData(const GrayscaleImage& packedTemplate, const GrayscaleImage& packedMask, string algorithmSignature)
{
	this->irisTemplate = packedTemplate;
	this->mask = packedMask;
	this->encoderSignature = algorithmSignature;
}

unsigned IrisTemplate::getValidBitCount() const
{
	unsigned total = this->mask.cols*this->mask.rows*8;			// *8 because it's the byte-packed representation
	unsigned nonzero = tools::countNonZeroBits(this->mask);
	return (100*nonzero)/total;
}
