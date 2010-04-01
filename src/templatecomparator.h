/*
 * templatecomparator.h
 *
 *  Created on: Jun 14, 2009
 *      Author: marcelo
 */

#pragma once

#include "common.h"
#include "iristemplate.h"
#include <vector>

class TemplateComparator {
public:
	TemplateComparator(int nRots=20, int rotStep=2);
	TemplateComparator(const IrisTemplate& irisTemplate, int nRots=20, int rotStep=2);
	virtual ~TemplateComparator();

	void setSrcTemplate(const IrisTemplate& irisTemplate);
	double compare(const IrisTemplate& otherTemplate);
	
	// For "a contrario" matching
	std::vector<double> compareParts(const IrisTemplate& otherTemplate, int nParts = 4);

	std::vector<IrisTemplate> rotatedTemplates;

private:
	static void rotateMatrix(const Mat& src, Mat& dest, int step);

	Mat_<uint8_t> maskIntersection;
	Mat_<uint8_t> xorBuffer;

	double packedHammingDistance(const Mat_<uint8_t>& template1, const Mat_<uint8_t>& mask1, const Mat_<uint8_t>& template2, const Mat_<uint8_t>& mask2);

	int nRots, rotStep;

};
