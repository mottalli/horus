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
	IrisTemplate(const CvMat* binaryTemplate, const CvMat* binaryMask);

	IplImage* getTemplate(void) const;
	IplImage* getNoiseMask(void) const;

	virtual ~IrisTemplate();
private:
	CvMat* irisTemplate;
	CvMat* mask;
};

#endif /* IRISTEMPLATE_H_ */
