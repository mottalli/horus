#include <cutil_inline.h>
#include <vector>
#include <stdint.h>
#include <cassert>
#include <iostream>
#include "cudacommon.h"

using namespace std;

#define XOR(a, b, mask1, mask2) (((~a & b) | (a & ~b)) & mask1 & mask2)
#define MAX_ROTS 100

inline __device__ unsigned countNonZeroBits(uint32_t v)
{
	v = v - ((v >> 1) & 0x55555555);                    // reuse input as temporary
	v = (v & 0x33333333) + ((v >> 2) & 0x33333333);     // temp
	unsigned c = ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24; // count
	return c;
}

__global__ void doGPUMatchKernel(const uint8_t* rotatedTemplates, const uint8_t* rotatedMasks, size_t nRotatedTemplates, const GPUDatabase database, float* distances)
{
	__shared__ float hammingDistances[MAX_ROTS];

	unsigned templateIdx = blockIdx.x;
	
	size_t templateSize = database.templateWidth * database.templateHeight;
	size_t templateWords = templateSize / sizeof(uint32_t);
	
	// Cast from chars to words
	uint32_t* template_ = (uint32_t*)(rotatedTemplates + threadIdx.x*templateSize);
	uint32_t* mask = (uint32_t*)(rotatedMasks + threadIdx.x*templateSize);
	uint32_t* otherTemplate = (uint32_t*)(database.d_templates + templateIdx*templateSize);
	uint32_t* otherMask = (uint32_t*)(database.d_masks + templateIdx*templateSize);
	
	size_t nonZeroBits = 0, totalBits = 0;
	uint32_t word1, word2, mask1, mask2;
	for (size_t i = 0; i < templateWords; i++) {
		word1 = template_[i];
		word2 = otherTemplate[i];
		mask1 = mask[i];
		mask2 = otherMask[i];
		
		nonZeroBits += countNonZeroBits(XOR(word1, word2, mask1, mask2));
		totalBits += countNonZeroBits(mask1 & mask2);
	}
	
	hammingDistances[threadIdx.x] = float(nonZeroBits) / float(totalBits);
	
	__syncthreads();
	
	if (threadIdx.x == 0) {
		float minHD = 1.0;
		for (int i = 0; i < blockDim.x; i++) {
			minHD = min(minHD, hammingDistances[i]);
		}
		distances[templateIdx] = minHD;
	}
}

/**
 * Load the database in the GPU
 */
void loadDatabase(const vector<const uint8_t*>& templates, const vector<const uint8_t*>& masks, size_t templateWidth, size_t templateHeight, GPUDatabase* database)
{
	assert(templateWidth % 4 == 0);			// For casting to int32 in the GPU (4x speedup)
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
		cutilSafeCall(cudaMemcpy(database->d_masks + i*templateSize, masks[i], templateSize, cudaMemcpyHostToDevice));
	}
	cout << "Finished loading." << endl;
};

void doGPUMatch(const vector<const uint8_t*>& rotatedTemplates, const vector<const uint8_t*>& rotatedMasks, GPUDatabase* database)
{
	assert(rotatedTemplates.size() == rotatedMasks.size());
	assert(rotatedTemplates.size() < MAX_ROTS);

	unsigned timer;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
	
	// Load the rotated templates and masks to the GPU
	uint8_t *d_rotatedTemplates, *d_rotatedMasks;
	size_t templateSize = database->templateWidth * database->templateHeight;
	size_t bytes = rotatedTemplates.size() * templateSize;
	
	cutilSafeCall(cudaMalloc(&d_rotatedTemplates, bytes));
	cutilSafeCall(cudaMalloc(&d_rotatedMasks, bytes));
	for (size_t i = 0; i < rotatedTemplates.size(); i++) {
		cutilSafeCall(cudaMemcpy(d_rotatedTemplates + i*templateSize, rotatedTemplates[i], templateSize, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_rotatedMasks + i*templateSize, rotatedMasks[i], templateSize, cudaMemcpyHostToDevice));
	}
	
	// Output buffer in device
	float* d_distances;
	cutilSafeCall(cudaMalloc(&d_distances, database->numberOfTemplates*sizeof(float)));


	// Invoke the kernel
	dim3 blockSize(rotatedTemplates.size(), 1, 1);
	dim3 gridSize(database->numberOfTemplates, 1);
	
	doGPUMatchKernel<<<gridSize, blockSize>>>(
		d_rotatedTemplates,
		d_rotatedMasks,
		rotatedTemplates.size(),
		*database,
		d_distances
	);
	
	// Retrieve the result
	float* distances = new float[database->numberOfTemplates];
	cudaMemcpy(distances, d_distances, database->numberOfTemplates*sizeof(float), cudaMemcpyDeviceToHost);
	
	// Free the memory
	cutilSafeCall(cudaFree(d_rotatedTemplates));
	cutilSafeCall(cudaFree(d_rotatedMasks));
	cutilSafeCall(cudaFree(d_distances));
	
	
	cutStopTimer(timer);

	/*for (size_t i = 0; i < database->numberOfTemplates; i++) {
		cout << distances[i] << endl;
	}*/

	cout << "Tiempo: " << cutGetTimerValue(timer) / 1000.0 << " s." << endl;
};
