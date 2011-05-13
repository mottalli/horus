/*
 * templatecomparator.cpp
 *
 *  Created on: Jun 14, 2009
 *      Author: marcelo
 */

#include "templatecomparator.h"
#include <cmath>
#include <iostream>

int countNonZeroBits(const Mat& mat);

TemplateComparator::TemplateComparator(int nRots, int rotStep)
{
	this->nRots = nRots;
	this->rotStep = rotStep;
	this->minHDIdx = -1;
}

TemplateComparator::TemplateComparator(const IrisTemplate& irisTemplate, int nRots, int rotStep)
{
	this->nRots = nRots;
	this->rotStep = rotStep;

	this->setSrcTemplate(irisTemplate);
}

TemplateComparator::~TemplateComparator()
{
}

double TemplateComparator::compare(const IrisTemplate& otherTemplate)
{
	double minHD = 1.0;

	assert(this->irisTemplate.encoderSignature == otherTemplate.encoderSignature);		// Must be the same "type" of template

	//for (std::vector<IrisTemplate>::const_iterator it = this->rotatedTemplates.begin(); it != this->rotatedTemplates.end(); it++) {
	for (size_t i = 0; i < this->rotatedTemplates.size(); i++) {
		//const IrisTemplate& rotatedTemplate = (*it);
		const IrisTemplate& rotatedTemplate = this->rotatedTemplates[i];
		double hd = this->packedHammingDistance(rotatedTemplate.getPackedTemplate(), rotatedTemplate.getPackedMask(),
				otherTemplate.getPackedTemplate(), otherTemplate.getPackedMask());
		if (hd < minHD) {
			minHD = hd;
			this->minHDIdx = i;
		}
	}

	return minHD;
}

std::vector<double> TemplateComparator::compareParts(const IrisTemplate& otherTemplate, int nParts)
{
	// Note: the width of the template must be a multiple of 8*nParts (remember this->irisTemplate has packed bits)
	int templateWidth = this->rotatedTemplates[0].getPackedTemplate().cols;
	int templateHeight = this->rotatedTemplates[0].getPackedTemplate().rows;

	assert((templateWidth % nParts) == 0);
	
	int partWidth = templateWidth / nParts;
	std::vector<double> minHDs(nParts, 1.0);
	
	for (std::vector<IrisTemplate>::const_iterator it = this->rotatedTemplates.begin(); it != this->rotatedTemplates.end(); it++) {
		const IrisTemplate& rotatedTemplate = (*it);

		for (int p = 0; p < nParts; p++) {
			Rect r(p*partWidth, 0, partWidth, templateHeight);

			GrayscaleImage part1(rotatedTemplate.getPackedTemplate(), r);
			GrayscaleImage mask1(rotatedTemplate.getPackedMask(), r);
			GrayscaleImage part2(otherTemplate.getPackedTemplate(), r);
			GrayscaleImage mask2(otherTemplate.getPackedMask(), r);
			
			double hd = this->packedHammingDistance(part1, mask1, part2, mask2);
			minHDs[p] = std::min(minHDs[p], hd);
		}
	}
	
	return minHDs;
}

void TemplateComparator::setSrcTemplate(const IrisTemplate& irisTemplate)
{
	this->irisTemplate = irisTemplate;

	Mat unpackedTemplate = irisTemplate.getUnpackedTemplate();
	Mat unpackedMask = irisTemplate.getUnpackedMask();
	
	assert((unpackedTemplate.cols % 8) == 0);

	// Buffers for storing the rotated templates and masks
	Mat rotatedTemplate = unpackedTemplate.clone();
	Mat rotatedMask = unpackedMask.clone();
	
	assert(irisTemplate.getPackedTemplate().size() == irisTemplate.getPackedMask().size());

	int packedWidth = irisTemplate.getPackedMask().cols;
	int packedHeight = irisTemplate.getPackedMask().rows;
	this->maskIntersection.create(packedHeight, packedWidth);
	this->xorBuffer.create(packedHeight, packedWidth);

	this->rotatedTemplates.clear();
	this->rotatedTemplates.push_back(irisTemplate);

	for (int r = this->rotStep; r <= this->nRots; r += this->rotStep) {
		this->rotateMatrix(unpackedTemplate, rotatedTemplate, r);
		this->rotateMatrix(unpackedMask, rotatedMask, r);
		this->rotatedTemplates.push_back(IrisTemplate(rotatedTemplate, rotatedMask, irisTemplate.encoderSignature));

		this->rotateMatrix(unpackedTemplate, rotatedTemplate, -r);
		this->rotateMatrix(unpackedMask, rotatedMask, -r);
		this->rotatedTemplates.push_back(IrisTemplate(rotatedTemplate, rotatedMask, irisTemplate.encoderSignature));
	}
}

void TemplateComparator::rotateMatrix(const Mat& src, Mat& dest, int step)
{
	if (step == 0) {
		src.copyTo(dest);
	} else if (step < 0) {
		dest.create(src.size(), CV_8U);

		// Rotate left
		step = -step;

		Mat pieceSrc, pieceDest;
		pieceSrc = src(Rect(0, 0, step, src.rows));
		pieceDest = dest(Rect(src.cols-step, 0, step, src.rows));
		pieceSrc.copyTo(pieceDest);

		pieceSrc = src(Rect(step, 0, src.cols-step, src.rows));
		pieceDest = dest(Rect(0, 0, src.cols-step, src.rows));
		pieceSrc.copyTo(pieceDest);
	} else {
		dest.create(src.size(), CV_8U);

		// Rotate right
		Mat pieceSrc, pieceDest;

		pieceSrc = src(Rect(0, 0, src.cols-step, src.rows));
		pieceDest = dest(Rect(step, 0, src.cols-step, src.rows));
		pieceSrc.copyTo(pieceDest);

		pieceSrc = src(Rect(src.cols-step, 0, step, src.rows));
		pieceDest = dest(Rect(0, 0, step, src.rows));
		pieceSrc.copyTo(pieceDest);
	}
}

double TemplateComparator::packedHammingDistance(const GrayscaleImage& template1, const GrayscaleImage& mask1, const GrayscaleImage& template2, const GrayscaleImage& mask2)
{
	assert(template1.size() == mask1.size());
	assert(template2.size() == mask2.size());
	assert(template1.size() == template2.size());
	
	bitwise_and(mask1, mask2, this->maskIntersection);
	bitwise_xor(template1, template2, this->xorBuffer);
	bitwise_and(this->xorBuffer, this->maskIntersection, this->xorBuffer);

	int nonZeroBits = countNonZeroBits(xorBuffer);
	int validBits = countNonZeroBits(maskIntersection);

	/*//
	int template1ValidBits = countNonZeroBits(mask1);
	int template2ValidBits = countNonZeroBits(mask2);
	cout << 100.0*double(template1ValidBits)/double(template1.cols*template1.rows*8) << '%' << " ";
	cout << 100.0*double(template2ValidBits)/double(template1.cols*template1.rows*8) << '%' << " ";
	cout << 100.0*double(validBits)/double(maskIntersection.cols*maskIntersection.rows*8) << '%' << endl;
	//*/

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
int countNonZeroBits(const Mat& mat)
{
	assert(mat.depth() == CV_8U);

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

	for (int y = 0; y < mat.rows; y++) {
		const uint8_t* row = mat.ptr(y);
		int x;
		for (x = 0; x < mat.cols-3; x += 4) {		// Optimization: aligned to 4 byes, extracted from cvCountNonZero
			uint8_t val0 = row[x], val1 = row[x+1], val2 = row[x+2], val3 = row[x+3];
			res += nonZeroBits[val0] + nonZeroBits[val1] + nonZeroBits[val2] + nonZeroBits[val3];
		}
		for (; x < mat.cols; x++) {
			res += nonZeroBits[ row[x] ];
		}
	}
	return res;
}

const IrisTemplate& TemplateComparator::getBestRotatedTemplate()
{
	return this->rotatedTemplates[this->minHDIdx];
}
