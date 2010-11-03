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
	gpu::cleanupDatabase(&this->gpuDatabase);
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
		gpu::loadDatabase(this->templates, this->gpuDatabase);
		this->dirty = false;

		this->resultDistances = vector<double>(this->templates.size());
	}
	
	TemplateComparator comparator(irisTemplate, nRots, rotStep);
	gpu::doGPUAContrarioMatch(comparator, this->gpuDatabase, nParts, this->resultPartsDistances, this->matchingTime);
}

void IrisDatabaseCUDA::doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int), int nRots, int rotStep)
{
	if (this->dirty) {
		gpu::loadDatabase(this->templates, this->gpuDatabase);
		this->dirty = false;

		this->resultDistances = vector<double>(this->templates.size());
	}
	
	TemplateComparator comparator(irisTemplate, nRots, rotStep);
	gpu::doGPUMatch(comparator, this->gpuDatabase, this->resultDistances, this->matchingTime);
}
