/*
 * iristemplate.h
 *
 *  Created on: Jun 13, 2009
 *      Author: marcelo
 */

#pragma once

#include "common.h"
#include <string>

// Classes and methods that will be referenced later
class TemplateComparator;
class IrisTemplate;
namespace Serializer {
	IrisTemplate unserializeIrisTemplate(const std::string& serializedTemplate);
	std::string serializeIrisTemplate(const IrisTemplate& irisTemplate);
}

class IrisTemplate {
	friend class TemplateComparator;
	friend IrisTemplate Serializer::unserializeIrisTemplate(const std::string& serializedTemplate);
	friend std::string Serializer::serializeIrisTemplate(const IrisTemplate& irisTemplate);

public:
	IrisTemplate();
	IrisTemplate(const IrisTemplate& otherTemplate);
	IrisTemplate(const CvMat* binaryTemplate, const CvMat* binaryMask);

	// NOTE: Caller must release these!
	IplImage* getTemplateImage(void) const;
	IplImage* getNoiseMaskImage(void) const;
	CvMat* getUnpackedTemplate() const;
	CvMat* getUnpackedMask() const;

	inline const CvMat* getPackedTemplate() const { return this->irisTemplate; }
	inline const CvMat* getPackedMask() const { return this->mask; }

	IrisTemplate& operator=(const IrisTemplate& otherTemplate);

	virtual ~IrisTemplate();
protected:
	CvMat* irisTemplate;
	CvMat* mask;
};

