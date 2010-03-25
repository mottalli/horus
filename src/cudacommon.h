#pragma once

#include <stdint.h>
#include <vector>

using namespace std;

struct GPUDatabase {
	uint8_t* d_templates;
	uint8_t* d_masks;
	size_t templateWidth, templateHeight;
	size_t numberOfTemplates;

	GPUDatabase() {
		this->numberOfTemplates = 0;
		this->d_templates = this->d_masks = NULL;
	}
};

extern "C"
void loadDatabase(const vector<const uint8_t*>& templates, const vector<const uint8_t*>& masks, size_t templateWidth,
				  size_t templateHeight, GPUDatabase* database);

extern "C"
void cleanupDatabase(GPUDatabase* database);

extern "C"
void doGPUMatch(const vector<const uint8_t*>& rotatedTemplates, const vector<const uint8_t*>& rotatedMasks, GPUDatabase* database,
				vector<double>& resultDistances, double& matchingTime);

extern "C"
void doGPUAContrarioMatch(const vector<const uint8_t*>& rotatedTemplates, const vector<const uint8_t*>& rotatedMasks, GPUDatabase* database,
						  unsigned nParts, vector< vector<double> >& resultDistances, double& matchingTime);
