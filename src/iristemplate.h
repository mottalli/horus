#pragma once

#include <string>
#include "common.h"

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

	/**
	 * Returns a value between 0 and 100 with the percentage of valid (non-masked) bits in the template
	 */
	unsigned getValidBitCount() const;

	string encoderSignature;

	//TODO
	unsigned irisQuality;
	unsigned templateQuality;
protected:
	GrayscaleImage irisTemplate, mask;
};

}
