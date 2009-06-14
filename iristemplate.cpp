/*
 * iristemplate.cpp
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#include "iristemplate.h"
#include <iostream>

IrisTemplate::IrisTemplate()
{
	this->irisTemplate = NULL;
	this->mask = NULL;

}

IrisTemplate::IrisTemplate(const CvMat* binaryTemplate, const CvMat* binaryMask)
{
	assert(binaryTemplate->width % 8 == 0);
	assert(binaryMask->width% 8 == 0);

	this->irisTemplate = (CvMat*)cvClone(binaryTemplate);
	this->mask = (CvMat*)cvClone(binaryMask);
}

IrisTemplate::~IrisTemplate()
{
}

Image* IrisTemplate::getTemplate() const
{
	Image* foo = cvCreateImage(cvGetSize(this->irisTemplate), IPL_DEPTH_8U, 1);
	cvNormalize(this->irisTemplate, foo, 0, 255, CV_MINMAX);
	return foo;
}

Image* IrisTemplate::getNoiseMask() const
{
	Image* foo = cvCreateImage(cvGetSize(this->mask), IPL_DEPTH_8U, 1);
	cvNormalize(this->mask, foo, 0, 255, CV_MINMAX);
	return foo;
}
