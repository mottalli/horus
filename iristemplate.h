/*
 * iristemplate.h
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#ifndef IRISTEMPLATE_H_
#define IRISTEMPLATE_H_

#include "common.h"
#include <string>

class TemplateComparator;

class IrisTemplate {
	friend class TemplateComparator;
public:
	IrisTemplate();
	IrisTemplate(const IrisTemplate& otherTemplate);
	IrisTemplate(const CvMat* binaryTemplate, const CvMat* binaryMask);

	IplImage* getTemplateImage(void) const;
	IplImage* getNoiseMaskImage(void) const;
	CvMat* getUnpackedTemplate() const;
	CvMat* getUnpackedMask() const;

	inline const CvMat* getPackedTemplate() const { return this->irisTemplate; };
	inline const CvMat* getPackedMask() const { return this->mask; };

	IrisTemplate& operator=(const IrisTemplate& otherTemplate);

	std::string serialize() const;
	static IrisTemplate unserialize(const std::string& serializedTemplate);

	virtual ~IrisTemplate();
protected:
	CvMat* irisTemplate;
	CvMat* mask;
};

#endif /* IRISTEMPLATE_H_ */
