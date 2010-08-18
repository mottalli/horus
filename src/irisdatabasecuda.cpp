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
		loadDatabase(this->templates, this->gpuDatabase);
		this->dirty = false;

		this->resultDistances = vector<double>(this->templates.size());
	}
	
	TemplateComparator comparator(irisTemplate, nRots, rotStep);
	
	/*vector<const uint8_t*> rawRotatedTemplates(comparator.rotatedTemplates.size()), rawRotatedMasks(comparator.rotatedTemplates.size());
	for (size_t i = 0; i < comparator.rotatedTemplates.size(); i++) {
		assert(comparator.rotatedTemplates[i].getPackedTemplate().cols == this->gpuDatabase.templateWidth);
		assert(comparator.rotatedTemplates[i].getPackedTemplate().rows == this->gpuDatabase.templateHeight);
		
		rawRotatedTemplates[i] = comparator.rotatedTemplates[i].getPackedTemplate().data;
		rawRotatedMasks[i] = comparator.rotatedTemplates[i].getPackedMask().data;
	}
	
	
	doGPUAContrarioMatch(rawRotatedTemplates, rawRotatedMasks, &this->gpuDatabase, nParts, this->resultPartsDistances, this->matchingTime);*/
	doGPUAContrarioMatch(comparator, this->gpuDatabase, nParts, this->resultPartsDistances, this->matchingTime);
}

void IrisDatabaseCUDA::doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int), int nRots, int rotStep)
{
	if (this->dirty) {
		loadDatabase(this->templates, this->gpuDatabase);
		this->dirty = false;

		this->resultDistances = vector<double>(this->templates.size());
	}
	
	TemplateComparator comparator(irisTemplate, nRots, rotStep);
	doGPUMatch(comparator, this->gpuDatabase, this->resultDistances, this->matchingTime);
}
