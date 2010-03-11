#include "irisdatabasecuda.h"
#include "cudacommon.h"
#include "templatecomparator.h"

IrisDatabaseCUDA::IrisDatabaseCUDA()
{
	this->dirty = true;
}

void IrisDatabaseCUDA::addTemplate(int templateId, const IrisTemplate& irisTemplate)
{
	IrisDatabase::addTemplate(templateId, irisTemplate);
	this->dirty = true;
}

void IrisDatabaseCUDA::deleteTemplate(int templateId)
{
	IrisDatabase::deleteTemplate(templateId);
	this->dirty = true;
}

void IrisDatabaseCUDA::doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int), int nRots, int rotStep)
{
	this->clock.start();
	if (this->dirty) {
		vector<const uint8_t*> rawTemplates(this->templates.size()), rawMasks(this->templates.size());

		for (size_t i = 0; i < this->templates.size(); i++) {
			// Indirect way of assuring it's CV_8U
			assert(this->templates[i]->getPackedTemplate()->step == this->templates[i]->getPackedTemplate()->width);

			rawTemplates[i] = this->templates[i]->getPackedTemplate()->data.ptr;
			rawMasks[i] = this->templates[i]->getPackedMask()->data.ptr;
		}
		
		size_t packedWidth = this->templates[0]->getPackedTemplate()->width, packedHeight = this->templates[0]->getPackedTemplate()->height;

		loadDatabase(rawTemplates, rawMasks, packedWidth, packedHeight, &this->gpuDatabase);
		this->dirty = false;
	}
	
	TemplateComparator comparator(irisTemplate, nRots, rotStep);
	
	vector<const uint8_t*> rawRotatedTemplates(comparator.rotatedTemplates.size()), rawRotatedMasks(comparator.rotatedTemplates.size());
	for (size_t i = 0; i < comparator.rotatedTemplates.size(); i++) {
		assert(comparator.rotatedTemplates[i].getPackedTemplate()->width == this->gpuDatabase.templateWidth);
		assert(comparator.rotatedTemplates[i].getPackedTemplate()->height == this->gpuDatabase.templateHeight);
		
		rawRotatedTemplates[i] = comparator.rotatedTemplates[i].getPackedTemplate()->data.ptr;
		rawRotatedMasks[i] = comparator.rotatedTemplates[i].getPackedMask()->data.ptr;
	}
	
	doGPUMatch(rawRotatedTemplates, rawRotatedMasks, &this->gpuDatabase);
	
	this->clock.stop();
}
