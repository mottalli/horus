/*
 * templatecomparator.cpp
 *
 *  Created on: Jun 14, 2009
 *      Author: marcelo
 */

#include "templatecomparator.h"
#include "tools.h"

using namespace horus;

TemplateComparator::TemplateComparator(int nRots, int rotStep)
{
	this->nRots = nRots;
	this->rotStep = rotStep;
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

double TemplateComparator::compare(const IrisTemplate& otherTemplate) const
{
	double minHD = 1.0;

	assert(this->irisTemplate.encoderSignature == otherTemplate.encoderSignature);		// Must be the same "type" of template

	for (size_t i = 0; i < this->rotatedTemplates.size(); i++) {
		const IrisTemplate& rotatedTemplate = this->rotatedTemplates[i];
		double hd = TemplateComparator::packedHammingDistance(rotatedTemplate.getPackedTemplate(), rotatedTemplate.getPackedMask(),
				otherTemplate.getPackedTemplate(), otherTemplate.getPackedMask());
		if (hd < minHD) {
			minHD = hd;
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

			GrayscaleImage part1 = TemplateComparator::getPart(rotatedTemplate, p, nParts, false);
			GrayscaleImage mask1 = TemplateComparator::getPart(rotatedTemplate, p, nParts, true);
			GrayscaleImage part2 = TemplateComparator::getPart(otherTemplate, p, nParts, false);
			GrayscaleImage mask2 = TemplateComparator::getPart(otherTemplate, p, nParts, true);
			
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

	Mat1b maskIntersection, xorBuffer;

	bitwise_and(mask1, mask2, maskIntersection);
	bitwise_xor(template1, template2, xorBuffer);
	bitwise_and(xorBuffer, maskIntersection, xorBuffer);

	int nonZeroBits = horus::tools::countNonZeroBits(xorBuffer);
	int validBits = horus::tools::countNonZeroBits(maskIntersection);

	if (validBits == 0) {
		return 1.0;		// No bits to compare
	}

	assert(nonZeroBits <= validBits);

	return double(nonZeroBits)/double(validBits);
}

GrayscaleImage TemplateComparator::getComparationImage(const IrisTemplate& otherTemplate, bool showMask)
{
	//TODO
	return otherTemplate.getUnpackedTemplate();

	/*this->compare(otherTemplate);

	const IrisTemplate& t1 = this->getBestRotatedTemplate();
	const IrisTemplate& t2 = otherTemplate;

	GrayscaleImage i1 = t1.getUnpackedTemplate();
	GrayscaleImage i2 = t2.getUnpackedTemplate();

	GrayscaleImage res;
	i1.setTo(255, i1);
	i2.setTo(255, i2);

	bitwise_xor(i1, i2, res);			// Note: this sets to white the *different* bits
	bitwise_not(res, res);

	if (showMask) {
		GrayscaleImage m1 = t1.getUnpackedMask();
		GrayscaleImage m2 = t2.getUnpackedMask();
		m1.setTo(255, m1);
		m2.setTo(255, m2);
		bitwise_not(m1, m1);				// Set to 1 the invalid bits
		bitwise_not(m2, m2);
		res.setTo(128, m1);
		res.setTo(128, m2);
	}

	return res;*/
}
