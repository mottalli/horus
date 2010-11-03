#pragma once

#include "irisdatabase.h"
#include "cudacommon.h"
using namespace std;

class IrisDatabaseCUDA : public IrisDatabase
{
public:
    IrisDatabaseCUDA();
	virtual ~IrisDatabaseCUDA();

	void addTemplate(int templateId, const IrisTemplate& irisTemplate);
	void deleteTemplate(int templateId);

	void doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int) = NULL, int nRots=20, int rotStep=2);

protected:
	void calculatePartsDistances(const IrisTemplate& irisTemplate, int nParts, int nRots, int rotStep);
	
	bool dirty;
	GPUDatabase gpuDatabase;
};


