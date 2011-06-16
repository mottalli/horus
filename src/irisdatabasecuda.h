#pragma once

#include <cuda.h>

#include "irisdatabase.h"
#include "iristemplate.h"
#include "cudacommon.h"
#include "templatecomparator.h"

namespace horus {

class IrisDatabaseCUDA : public IrisDatabase
{
public:
    IrisDatabaseCUDA();
	virtual ~IrisDatabaseCUDA();

	virtual void addTemplate(int templateId, const IrisTemplate& irisTemplate);
	virtual void deleteTemplate(int templateId);
	virtual void doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int) = NULL, int nRots=20, int rotStep=2);
	virtual void doAContrarioMatch(const IrisTemplate& irisTemplate, size_t nParts=4, void (*statusCallback)(int) = NULL, int nRots=20, int rotStep=2);

protected:
	virtual void calculatePartsDistances(const IrisTemplate& irisTemplate, int nParts, int nRots, int rotStep);

	// Functions to interact with GPU
	void uploadDBToDevice();
	void cleanupDB();
	std::pair<uint8_t*, uint8_t*> uploadRotatedTemplates(const vector<IrisTemplate>& rotatedTemplates);
	
	bool dirty;
	GPUDatabase gpuDatabase;
};


}
