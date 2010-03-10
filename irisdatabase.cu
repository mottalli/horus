#include <cutil_inline.h>
#include <vector>
#include <stdint.h>
#include <iostream>
#include "cudacommon.h"

using namespace std;

#define XOR(a, b, mask1, mask2) (((~a & b) | (a & ~b)) & mask1 & mask2)

inline __device__ unsigned countNonZeroBits(uint32_t v)
{
	v = v - ((v >> 1) & 0x55555555);                    // reuse input as temporary
	v = (v & 0x33333333) + ((v >> 2) & 0x33333333);     // temp
	unsigned c = ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24; // count
	return c;
}

__global__ void doGPUMatchKernel(const uint8_t* template_, const uint8_t* mask, const uint8_t* templates, const uint8_t* masks, int templateWidth, int templateHeight, int nRots, int rotStep, float* distances)
{
	int rot = (threadIdx.x == 0 ? 0 : -nRots + (threadIdx.x-1)*rotStep);
	unsigned templateIdx = blockIdx.x;
	int templateSize = templateWidth*templateHeight;
	const uint8_t* otherTemplate = templates + templateIdx*templateSize;
	const uint8_t* otherMask = masks + templateIdx*templateSize;

	unsigned nonZeroBits = 0, totalBits = 0;
	
	int x0 = (templateWidth + rot) % templateWidth;

	uint8_t byte1, byte2, mask1, mask2;
	for (size_t y = 0; y < templateHeight; y++) {
		for (size_t x = 0; x < templateWidth; x++) {
			byte1 = template_[y*templateWidth + ((x0+x) % templateWidth)];
			mask1 = mask[y*templateWidth + ((x0+x) % templateWidth)];
		
			byte2 = otherTemplate[y*templateWidth+x];
			mask2 = otherMask[y*templateWidth+x];
			
			nonZeroBits += countNonZeroBits(XOR(byte1, byte2, mask1, mask2));
			totalBits += countNonZeroBits(mask1 & mask2);
		}
	}
	
	__syncthreads();
	
	if (threadIdx.x == 0) {
		distances[templateIdx] = float(nonZeroBits)/float(totalBits);
	}
}

/**
 * Load the database in the GPU
 */
void loadDatabase(vector<uint8_t*> templates, vector<uint8_t*> masks, size_t templateWidth, size_t templateHeight, GPUDatabase* database)
{
	size_t templateSize = templateWidth*templateHeight;

	database->templateWidth = templateWidth;
	database->templateHeight = templateHeight;
	database->numberOfTemplates = templates.size();

	size_t bytes = templates.size()*templateSize;
	cutilSafeCall(cudaMalloc(&database->d_templates, bytes));
	cutilSafeCall(cudaMalloc(&database->d_masks, bytes));

	// Load each individual template in a contiguous chunk of GPU memory
   cout << "Loading templates to GPU..." << endl;
	for (size_t i = 0; i < templates.size(); i++) {
		cutilSafeCall(cudaMemcpy(database->d_templates + i*templateSize, templates[i], templateSize, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(database->d_masks+ i*templateSize, masks[i], templateSize, cudaMemcpyHostToDevice));
	}
	cout << "Finished loading." << endl;
};

void doGPUMatch(uint8_t* template_, uint8_t* mask, GPUDatabase* database, int nRots, int rotStep)
{
	int blockx = 2*(nRots/rotStep)+1;
	dim3 blockSize(blockx, 1, 1);
	dim3 gridSize(database->numberOfTemplates, 1);
	size_t templateSize = database->templateWidth*database->templateHeight;

	float* d_distances;
	cutilSafeCall(cudaMalloc(&d_distances, database->numberOfTemplates*sizeof(float)));

	uint8_t* d_template;
	cutilSafeCall(cudaMalloc(&d_template, templateSize));
	cutilSafeCall(cudaMemcpy(d_template, template_, templateSize, cudaMemcpyHostToDevice));

	uint8_t* d_mask;
	cutilSafeCall(cudaMalloc(&d_mask, templateSize));
	cutilSafeCall(cudaMemcpy(d_mask, mask, templateSize, cudaMemcpyHostToDevice));

	cout << "Invoking kernel..." << endl;
	doGPUMatchKernel<<<gridSize, blockSize>>>(
			d_template,
			d_mask,
			database->d_templates,
			database->d_masks,
			database->templateWidth,
			database->templateHeight,
			nRots,
			rotStep,
			d_distances
		);

	cout << "End" << endl;

	float* distances = new float[database->numberOfTemplates];
	cudaMemcpy(distances, d_distances, database->numberOfTemplates*sizeof(float), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < database->numberOfTemplates; i++) {
		cout << distances[i] << endl;
	}

	cutilSafeCall(cudaFree(d_distances));
	cutilSafeCall(cudaFree(d_mask));
	cutilSafeCall(cudaFree(d_template));
};
