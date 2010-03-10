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
void loadDatabase(const vector<const uint8_t*>& templates, const vector<const uint8_t*>& masks, size_t templateWidth,
				  size_t templateHeight, GPUDatabase* database);

extern "C"
void doGPUMatch(const vector<const uint8_t*>& rotatedTemplates, const vector<const uint8_t*>& rotatedMasks, GPUDatabase* database);
