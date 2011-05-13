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
	//friend class TemplateComparator;
	friend IrisTemplate Serializer::unserializeIrisTemplate(const std::string& serializedTemplate);
	friend std::string Serializer::serializeIrisTemplate(const IrisTemplate& irisTemplate);

public:
	IrisTemplate();
	IrisTemplate(const IrisTemplate& otherTemplate);
	IrisTemplate(const GrayscaleImage& binaryTemplate, const GrayscaleImage& binaryMask, string algorithmSignature);
	virtual ~IrisTemplate();

	GrayscaleImage getTemplateImage() const;
	GrayscaleImage getUnpackedTemplate() const;
	GrayscaleImage getUnpackedMask() const;

	const GrayscaleImage& getPackedTemplate() const { return this->irisTemplate; };
	const GrayscaleImage& getPackedMask() const { return this->mask; };

	IrisTemplate& operator=(const IrisTemplate& otherTemplate);

	string encoderSignature;
protected:
	GrayscaleImage irisTemplate;
	GrayscaleImage mask;
};

