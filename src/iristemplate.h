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
	IrisTemplate(const Mat& binaryTemplate, const Mat& binaryMask);

	Mat getTemplateImage() const;
	Mat getNoiseMaskImage() const;
	Mat getUnpackedTemplate() const;
	Mat getUnpackedMask() const;

	const Mat_<uint8_t>& getPackedTemplate() const { return this->irisTemplate; };
	const Mat_<uint8_t>& getPackedMask() const { return this->mask; };

	IrisTemplate& operator=(const IrisTemplate& otherTemplate);

	virtual ~IrisTemplate();
protected:
	Mat_<uint8_t> irisTemplate;
	Mat_<uint8_t> mask;
};

