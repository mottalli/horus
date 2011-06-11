#pragma once

#include "common.h"
#include <string>

namespace horus {

class IrisTemplate {
public:
	IrisTemplate();
	IrisTemplate(const IrisTemplate& otherTemplate);
	IrisTemplate& operator=(const IrisTemplate& otherTemplate);
	IrisTemplate(const GrayscaleImage& binaryTemplate, const GrayscaleImage& binaryMask, string algorithmSignature);
	virtual ~IrisTemplate();

	GrayscaleImage getTemplateImage() const;
	GrayscaleImage getUnpackedTemplate() const;
	GrayscaleImage getUnpackedMask() const;

	const GrayscaleImage& getPackedTemplate() const { return this->irisTemplate; }
	const GrayscaleImage& getPackedMask() const { return this->mask; }

	void setPackedData(const GrayscaleImage& packedTemplate, const GrayscaleImage& packedMask, string algorithmSignature);

	string encoderSignature;
protected:
	GrayscaleImage irisTemplate, mask;
};

}
