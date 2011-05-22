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
}

IrisTemplate::IrisTemplate(const GrayscaleImage& binaryTemplate, const GrayscaleImage& binaryMask, string encoderSignature_)
{
	assert(binaryTemplate.size() == binaryMask.size());
	assert(binaryTemplate.depth() == CV_8U);
	assert(binaryMask.depth() == CV_8U);
	assert(binaryTemplate.channels() == 1);
	assert(binaryMask.channels() == 1);

	assert(binaryTemplate.cols % 8 == 0);

	Tools::packBits(binaryTemplate, this->irisTemplate);
	Tools::packBits(binaryMask, this->mask);

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
	Tools::unpackBits(this->irisTemplate, imgTemplate, 255);
	Tools::unpackBits(this->mask, imgMask, 255);

	bitwise_not(imgMask, imgMask);			// Hacky way to NOT the template
	imgTemplate.setTo(127, imgMask);

	return imgTemplate;
}

GrayscaleImage IrisTemplate::getUnpackedTemplate() const
{
	GrayscaleImage unpacked;
	Tools::unpackBits(this->irisTemplate, unpacked);
	return unpacked;
}

GrayscaleImage IrisTemplate::getUnpackedMask() const
{
	GrayscaleImage unpacked;
	Tools::unpackBits(this->mask, unpacked);
	return unpacked;
}

void IrisTemplate::setPackedData(const GrayscaleImage& packedTemplate, const GrayscaleImage& packedMask, string algorithmSignature)
{
	this->irisTemplate = packedTemplate;
	this->mask = packedMask;
	this->encoderSignature = algorithmSignature;
}
