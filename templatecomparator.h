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

	static void rotateMatrix(const CvMat* src, CvMat* dest, int step);

	struct {
		CvMat* maskIntersection;
		CvMat* xorBuffer;
	} buffers;

private:
	double packedHammingDistance(const CvMat* template1, const CvMat* mask1, const CvMat* template2, const CvMat* mask2);

	int nRots, rotStep;

	std::vector<IrisTemplate> rotatedTemplates;
};
