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

	const IrisTemplate& getBestRotatedTemplate();

	GrayscaleImage getComparationImage();

private:
	static void rotateMatrix(const Mat& src, Mat& dest, int step);
	IrisTemplate irisTemplate;

	GrayscaleImage maskIntersection;
	GrayscaleImage xorBuffer;

	double packedHammingDistance(const GrayscaleImage& template1, const GrayscaleImage& mask1, const GrayscaleImage& template2, const GrayscaleImage& mask2);

	int nRots, rotStep;
	int minHDIdx;
};
