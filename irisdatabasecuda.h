#pragma once

#include "irisdatabase.h"
#include "cudacommon.h"
using namespace std;

class IrisDatabaseCUDA : public IrisDatabase
{
public:
    IrisDatabaseCUDA();
	void addTemplate(int templateId, const IrisTemplate& irisTemplate);
	void deleteTemplate(int templateId);

	void doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int) = NULL, int nRots=20, int rotStep=2);

protected:
	bool dirty;
	GPUDatabase gpuDatabase;
};


