/*
 * templatecomparator.cpp
 *
 *  Created on: Jun 14, 2009
 *      Author: marcelo
 */

#include "templatecomparator.h"
#include <cmath>
#include <iostream>

int countNonZeroBits(const CvMat* mat);

TemplateComparator::TemplateComparator(int nRots, int rotStep) :
	nRots(nRots), rotStep(rotStep)
{
	this->buffers.maskIntersection = NULL;
	this->buffers.xorBuffer = NULL;
}

TemplateComparator::TemplateComparator(const IrisTemplate& irisTemplate, int nRots, int rotStep) :
	nRots(nRots), rotStep(rotStep)
{
	this->buffers.maskIntersection = NULL;
	this->buffers.xorBuffer = NULL;
	this->setSrcTemplate(irisTemplate);
}

TemplateComparator::~TemplateComparator()
{
	if (this->buffers.maskIntersection != NULL) {
		cvReleaseMat(&this->buffers.maskIntersection);
		cvReleaseMat(&this->buffers.xorBuffer);
	}
}

double TemplateComparator::compare(const IrisTemplate& otherTemplate)
{
	double minHD = 1.0;

	assert(this->buffers.maskIntersection != NULL);
	assert(this->buffers.xorBuffer != NULL);

	for (std::vector<IrisTemplate>::const_iterator it = this->rotatedTemplates.begin(); it != this->rotatedTemplates.end(); it++) {
		const IrisTemplate& rotatedTemplate = (*it);
		double hd = this->packedHammingDistance(rotatedTemplate.getPackedTemplate(), rotatedTemplate.getPackedMask(),
				otherTemplate.getPackedTemplate(), otherTemplate.getPackedMask());
		minHD = std::min(hd, minHD);
	}

	return minHD;
}

std::vector<double> TemplateComparator::compareParts(const IrisTemplate& otherTemplate, int nParts)
{
	// Note: the width of the template must be a multiple of 8*nParts (remember this->irisTemplate has packed bits)
	int templateWidth = this->rotatedTemplates[0].getPackedTemplate()->width;
	int templateHeight = this->rotatedTemplates[0].getPackedTemplate()->height;

	assert((templateWidth % nParts) == 0);
	
	int partWidth = templateWidth / nParts;
	std::vector<double> minHDs(nParts, 1.0);
	
	CvMat part1, mask1, part2, mask2;
	
	for (std::vector<IrisTemplate>::const_iterator it = this->rotatedTemplates.begin(); it != this->rotatedTemplates.end(); it++) {
		const IrisTemplate& rotatedTemplate = (*it);
		
		for (int p = 0; p < nParts; p++) {
			CvRect r = cvRect(p*partWidth, 0, partWidth, templateHeight);
			cvGetSubRect(rotatedTemplate.getPackedTemplate(), &part1, r);
			cvGetSubRect(rotatedTemplate.getPackedMask(), &mask1, r);
			cvGetSubRect(otherTemplate.getPackedTemplate(), &part2, r);
			cvGetSubRect(otherTemplate.getPackedMask(), &mask2, r);
			
			double hd = this->packedHammingDistance(&part1, &mask1, &part2, &mask2);
			minHDs[p] = std::min(minHDs[p], hd);
		}
	}
	
	return minHDs;
}

void TemplateComparator::setSrcTemplate(const IrisTemplate& irisTemplate)
{

	CvMat* unpackedTemplate = irisTemplate.getUnpackedTemplate();
	CvMat* unpackedMask = irisTemplate.getUnpackedMask();
	
	//assert((unpackedTemplate->cols % 8) == 0);

	// Buffers for storing the rotated templates and masks
	CvMat* rotatedTemplate = cvCloneMat(unpackedTemplate);
	CvMat* rotatedMask = cvCloneMat(unpackedMask);
	
	if (this->buffers.maskIntersection != NULL) {
		cvReleaseMat(&this->buffers.maskIntersection);
		cvReleaseMat(&this->buffers.xorBuffer);
	}
	
	assert(SAME_SIZE(irisTemplate.getPackedMask(), irisTemplate.getPackedTemplate()));

	int packedWidth = irisTemplate.getPackedMask()->width;
	int packedHeight = irisTemplate.getPackedMask()->height;
	this->buffers.maskIntersection = cvCreateMat(packedHeight, packedWidth, CV_8U);
	this->buffers.xorBuffer = cvCreateMat(packedHeight, packedWidth, CV_8U);

	this->rotatedTemplates.clear();
	this->rotatedTemplates.push_back(irisTemplate);

	for (int r = this->rotStep; r <= this->nRots; r += this->rotStep) {
		this->rotateMatrix(unpackedTemplate, rotatedTemplate, r);
		this->rotateMatrix(unpackedMask, rotatedMask, r);
		this->rotatedTemplates.push_back(IrisTemplate(rotatedTemplate, rotatedMask));

		this->rotateMatrix(unpackedTemplate, rotatedTemplate, -r);
		this->rotateMatrix(unpackedMask, rotatedMask, -r);
		this->rotatedTemplates.push_back(IrisTemplate(rotatedTemplate, rotatedMask));
	}

	cvReleaseMat(&unpackedTemplate);
	cvReleaseMat(&unpackedMask);
	cvReleaseMat(&rotatedTemplate);
	cvReleaseMat(&rotatedMask);
}

void TemplateComparator::rotateMatrix(const CvMat* src, CvMat* dest, int step)
{
	assert(SAME_SIZE(src, dest));

	if (step == 0) {
		cvCopy(src, dest);
	} else if (step < 0) {
		// Rotate left
		step = -step;
		CvMat pieceSrc, pieceDest;

		cvGetSubRect(src, &pieceSrc, cvRect(0, 0, step, src->height));
		cvGetSubRect(dest, &pieceDest, cvRect(src->width-step, 0, step, src->height));
		cvCopy(&pieceSrc, &pieceDest);

		cvGetSubRect(src, &pieceSrc, cvRect(step, 0, src->width-step, src->height));
		cvGetSubRect(dest, &pieceDest, cvRect(0, 0, src->width-step, src->height));
		cvCopy(&pieceSrc, &pieceDest);

	} else {
		// Rotate right
		CvMat pieceSrc, pieceDest;

		cvGetSubRect(src, &pieceSrc, cvRect(0, 0, src->width-step, src->height));
		cvGetSubRect(dest, &pieceDest, cvRect(step, 0, src->width-step, src->height));
		cvCopy(&pieceSrc, &pieceDest);

		cvGetSubRect(src, &pieceSrc, cvRect(src->width-step, 0, step, src->height));
		cvGetSubRect(dest, &pieceDest, cvRect(0, 0, step, src->height));
		cvCopy(&pieceSrc, &pieceDest);
	}
}

double TemplateComparator::packedHammingDistance(const CvMat* template1, const CvMat* mask1, const CvMat* template2, const CvMat* mask2)
{
	assert(SAME_SIZE(template1, mask1));
	assert(SAME_SIZE(template2, mask2));
	assert(SAME_SIZE(template1, template2));
	//assert(SAME_SIZE(mask1, this->buffers.maskIntersection));
	//assert(SAME_SIZE(template1, this->buffers.xorBuffer));
	
	// This is done so we can use the same buffer in the call to compareParts()
	CvMat maskIntersection, xorBuffer;
	cvGetSubRect(this->buffers.maskIntersection, &maskIntersection, cvRect(0, 0, mask1->width, mask1->height));
	cvGetSubRect(this->buffers.xorBuffer, &xorBuffer, cvRect(0, 0, template1->width, template1->height));

	cvAnd(mask1, mask2, &maskIntersection);
	cvXor(template1, template2, &xorBuffer);
	cvAnd(&xorBuffer, &maskIntersection, &xorBuffer);

	int nonZeroBits = countNonZeroBits(&xorBuffer);
	int validBits = countNonZeroBits(&maskIntersection);

	if (validBits == 0) {
		return 1.0;		// No bits to compare
	}

	assert(nonZeroBits <= validBits);

	return double(nonZeroBits)/double(validBits);
}


/**
 * 0000 0
 * 0001 1
 * -------
 * 0010 1 = 0+1
 * 0011 2 = 1+1
 * -------
 * 0100 1 = 0+1
 * 0101 2 = 1+1
 * 0110 2 = 0+1+1
 * 0111 3 = 1+1+1
 * -------
 * 1000 1 = 0+1
 * 1001 2 = 1+1
 * 1010 2 = 0+1+1
 * 1011 3 = 1+1+1
 * 1100 2 = 0+1+1
 * 1101 3 = 1+1+1
 * 1110 3 = 0+1+1+1
 * 1111 4 = 1+1+1+1
 * --------
 * etc...
 */
int countNonZeroBits(const CvMat* mat)
{
	//assert(mat->type == CV_8U);		// For some reason this doesn't work! (bug in OpenCV? mat->type gets some random value)

	static bool initialized = false;
	static int nonZeroBits[256];		// Note: this could be hard-coded but it's too long and the algorithm to calculate it
										// is quite simple and we only do it once
	if (!initialized) {
		// Calculate the number of bits set to 1 on an 8-bit value
		nonZeroBits[0] = 0;
		nonZeroBits[1] = 1;

		// Sorry about non-meaningful variable names :)
		int p = 2;
		while (p < 256) {
			for (int q = 0; q < p; q++) {
				nonZeroBits[p+q] = nonZeroBits[q]+1;
			}
			p = 2*p;
		}

		initialized = true;
	}

	int res = 0;

	for (int y = 0; y < mat->rows; y++) {
		uint8_t* row = (uint8_t*)(mat->data.ptr) + y*mat->step;
		int x;
		for (x = 0; x < mat->cols-3; x += 4) {		// Optimization: aligned to 4 byes, extracted from cvCountNonZero
			uint8_t val0 = row[x], val1 = row[x+1], val2 = row[x+2], val3 = row[x+3];
			res += nonZeroBits[val0] + nonZeroBits[val1] + nonZeroBits[val2] + nonZeroBits[val3];
		}
		for (; x < mat->cols; x++) {
			res += nonZeroBits[ row[x] ];
		}
	}
	return res;
}
