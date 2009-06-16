/*
 * templatecomparator.h
 *
 *  Created on: Jun 14, 2009
 *      Author: marcelo
 */

#ifndef TEMPLATECOMPARATOR_H_
#define TEMPLATECOMPARATOR_H_

#include "common.h"
#include "iristemplate.h"
#include <vector>

class TemplateComparator {
public:
	TemplateComparator(int nRots = 20, int rotStep = 2);
	TemplateComparator(const IrisTemplate& irisTemplate, int nRots = 20, int rotStep = 2);
	virtual ~TemplateComparator();

	void setSrcTemplate(const IrisTemplate& irisTemplate);
	double compare(const IrisTemplate& otherTemplate);

	struct {
		CvMat* maskIntersection;
		CvMat* xorBuffer;
	} buffers;

private:
	double hammingDistance(const CvMat* template1, const CvMat* mask1, const CvMat* template2, const CvMat* mask2);
	void rotateMatrix(const CvMat* src, CvMat* dest, int step);

	int nRots, rotStep;

	std::vector<IrisTemplate> rotatedTemplates;
};

#endif /* TEMPLATECOMPARATOR_H_ */
