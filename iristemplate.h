/*
 * iristemplate.h
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#ifndef IRISTEMPLATE_H_
#define IRISTEMPLATE_H_

#include "common.h"

class IrisTemplate {
public:
	IrisTemplate();
	IrisTemplate(const IrisTemplate& otherTemplate);
	IrisTemplate(const CvMat* binaryTemplate, const CvMat* binaryMask);

	IplImage* getTemplate(void) const;
	IplImage* getNoiseMask(void) const;

	IrisTemplate& operator=(const IrisTemplate& otherTemplate);

	virtual ~IrisTemplate();
private:
	CvMat* irisTemplate;
	CvMat* mask;

	void packBits(const CvMat* src, CvMat* dest) const;
	void unpackBits(const CvMat* src, CvMat* dest, int trueval = 1) const;
};

#endif /* IRISTEMPLATE_H_ */
