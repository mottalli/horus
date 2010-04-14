#include "irisdatabasecuda.h"
#include "cudacommon.h"
#include "templatecomparator.h"

IrisDatabaseCUDA::IrisDatabaseCUDA()
{
	this->dirty = true;
	this->matchingTime = 0;
}

IrisDatabaseCUDA::~IrisDatabaseCUDA()
{
	cleanupDatabase(&this->gpuDatabase);
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

void IrisDatabaseCUDA::calculatePartsDistances(const IrisTemplate& irisTemplate, int nParts, int nRots, int rotStep)
{
	size_t n = this->templates.size();

	assert(this->resultPartsDistances.size() == nParts);
	assert(this->resultPartsDistances[0].size() == n);

	if (this->dirty) {
		vector<const uint8_t*> rawTemplates(this->templates.size()), rawMasks(this->templates.size());

		for (size_t i = 0; i < this->templates.size(); i++) {
			// Indirect way of assuring it's CV_8U
			const Mat& packedTemplate = this->templates[i]->getPackedTemplate();
			const Mat& packedMask = this->templates[i]->getPackedMask();
			assert(packedTemplate.isContinuous() && packedMask.isContinuous());
			assert(packedTemplate.channels() == 1 && packedMask.channels() == 1);
			assert(packedTemplate.type() == CV_8U && packedMask.type() == CV_8U);

			rawTemplates[i] = packedTemplate.data;
			rawMasks[i] = packedMask.data;
		}
		
		size_t packedWidth = this->templates[0]->getPackedTemplate().cols, packedHeight = this->templates[0]->getPackedTemplate().rows;

		loadDatabase(rawTemplates, rawMasks, packedWidth, packedHeight, &this->gpuDatabase);
		this->dirty = false;

		this->resultDistances = vector<double>(this->templates.size());
	}
	
	TemplateComparator comparator(irisTemplate, nRots, rotStep);
	
	vector<const uint8_t*> rawRotatedTemplates(comparator.rotatedTemplates.size()), rawRotatedMasks(comparator.rotatedTemplates.size());
	for (size_t i = 0; i < comparator.rotatedTemplates.size(); i++) {
		assert(comparator.rotatedTemplates[i].getPackedTemplate().cols == this->gpuDatabase.templateWidth);
		assert(comparator.rotatedTemplates[i].getPackedTemplate().rows == this->gpuDatabase.templateHeight);
		
		rawRotatedTemplates[i] = comparator.rotatedTemplates[i].getPackedTemplate().data;
		rawRotatedMasks[i] = comparator.rotatedTemplates[i].getPackedMask().data;
	}
	
	
	doGPUAContrarioMatch(rawRotatedTemplates, rawRotatedMasks, &this->gpuDatabase, nParts, this->resultPartsDistances, this->matchingTime);
}

void IrisDatabaseCUDA::doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int), int nRots, int rotStep)
{
	if (this->dirty) {
		vector<const uint8_t*> rawTemplates(this->templates.size()), rawMasks(this->templates.size());

		for (size_t i = 0; i < this->templates.size(); i++) {
			// Indirect way of assuring it's CV_8U
			const Mat& packedTemplate = this->templates[i]->getPackedTemplate();
			const Mat& packedMask = this->templates[i]->getPackedMask();
			assert(packedTemplate.isContinuous() && packedMask.isContinuous());
			assert(packedTemplate.channels() == 1 && packedMask.channels() == 1);
			assert(packedTemplate.type() == CV_8U && packedMask.type() == CV_8U);

			rawTemplates[i] = packedTemplate.data;
			rawMasks[i] = packedMask.data;
		}
		
		size_t packedWidth = this->templates[0]->getPackedTemplate().cols, packedHeight = this->templates[0]->getPackedTemplate().rows;

		loadDatabase(rawTemplates, rawMasks, packedWidth, packedHeight, &this->gpuDatabase);
		this->dirty = false;

		this->resultDistances = vector<double>(this->templates.size());
	}
	
	TemplateComparator comparator(irisTemplate, nRots, rotStep);
	
	vector<const uint8_t*> rawRotatedTemplates(comparator.rotatedTemplates.size()), rawRotatedMasks(comparator.rotatedTemplates.size());
	for (size_t i = 0; i < comparator.rotatedTemplates.size(); i++) {
		assert(comparator.rotatedTemplates[i].getPackedTemplate().cols == this->gpuDatabase.templateWidth);
		assert(comparator.rotatedTemplates[i].getPackedTemplate().rows == this->gpuDatabase.templateHeight);
		
		rawRotatedTemplates[i] = comparator.rotatedTemplates[i].getPackedTemplate().data;
		rawRotatedMasks[i] = comparator.rotatedTemplates[i].getPackedMask().data;
	}
	
	doGPUMatch(rawRotatedTemplates, rawRotatedMasks, &this->gpuDatabase, this->resultDistances, this->matchingTime);
}
