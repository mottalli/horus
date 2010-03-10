#pragma once

#include <stdint.h>
#include <vector>

using namespace std;

typedef struct {
	uint8_t* d_templates;
	uint8_t* d_masks;
	size_t templateWidth, templateHeight;
	size_t numberOfTemplates;
} GPUDatabase;

extern "C"
void loadDatabase(vector<uint8_t*> templates, vector<uint8_t*> masks, size_t templateWidth,
				  size_t templateHeight, GPUDatabase* database);

extern "C"
void doGPUMatch(uint8_t* template_, uint8_t* mask, GPUDatabase* database, int nRots, int rotStep);
