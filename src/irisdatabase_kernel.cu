#include <vector>
#include <stdint.h>
#include <cassert>
#include <iostream>
#include <stdio.h>
#include "cudacommon.h"
#include "clock.h"

using namespace std;

#define XOR(a, b, mask1, mask2) ((a ^ b) & mask1 & mask2)

__global__ void doGPUMatchKernel(const uint8_t* rotatedTemplates, const uint8_t* rotatedMasks, size_t nRotatedTemplates, const GPUDatabase database, float* distances)
{
	__shared__ float hammingDistances[MAX_ROTS];

	unsigned templateIdx = blockIdx.x;
	
	if (templateIdx > database.numberOfTemplates) {
		return;
	}
	
	size_t templateSize = database.templateWidth * database.templateHeight;
	size_t templateWords = templateSize / 4;			// 4 == sizeof(uint32_t);
	
	// Cast from chars to words
	uint32_t* rotatedTemplate = (uint32_t*)(rotatedTemplates + threadIdx.x*templateSize);
	uint32_t* rotatedMask = (uint32_t*)(rotatedMasks + threadIdx.x*templateSize);
	uint32_t* otherTemplate = (uint32_t*)(database.d_templates + templateIdx*templateSize);
	uint32_t* otherMask = (uint32_t*)(database.d_masks + templateIdx*templateSize);
	
	size_t nonZeroBits = 0, totalBits = 0;
	uint32_t word1, mask1;
	__shared__ uint32_t word2, mask2;
	
	for (size_t i = 0; i < templateWords; i++) {
		word1 = rotatedTemplate[i];
		mask1 = rotatedMask[i];
		if (threadIdx.x == 0) {
			word2 = otherTemplate[i];
			mask2 = otherMask[i];
		}
		__syncthreads();
		
		// __popc(x) returns the number of bits that are set to 1 in the binary representation of 32-bit integer parameter x.
		uint32_t x = XOR(word1, word2, mask1, mask2);
		nonZeroBits += __popc(x);
		totalBits += __popc(mask1 & mask2);
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

__global__ void doGPUAContrarioMatchKernel(const uint8_t* rotatedTemplates, const uint8_t* rotatedMasks, size_t nRotatedTemplates, const GPUDatabase database, float* distances)
{
	__shared__ float hammingDistances[MAX_PARTS][MAX_ROTS];

	unsigned templateIdx = blockIdx.x;
	unsigned part = threadIdx.y;
	unsigned nParts = blockDim.y;

	//assert((database.templateWidth % 4) == 0);

	size_t templateSize = database.templateWidth * database.templateHeight;

	uint32_t* rotatedTemplate = (uint32_t*)(rotatedTemplates + threadIdx.x*templateSize);
	uint32_t* rotatedMask = (uint32_t*)(rotatedMasks + threadIdx.x*templateSize);
	uint32_t* otherTemplate = (uint32_t*)(database.d_templates + templateIdx*templateSize);
	uint32_t* otherMask = (uint32_t*)(database.d_masks + templateIdx*templateSize);

	unsigned widthRows = database.templateWidth / 4;		// Width of the template in 32-bit words
	unsigned partWidthWords = ceil(float(widthRows)/float(nParts));		// Width of the part in 32-bit words

	size_t nonZeroBits = 0, totalBits = 0;
	uint32_t word1, mask1;
	__shared__ uint32_t words2[MAX_ROTS], masks2[MAX_ROTS], word2, mask2;

	unsigned w0row = floor((float(widthRows)/float(nParts))*float(part));			// Offset of the first word in the part for each row

	unsigned idx;
	for (unsigned row = 0; row < database.templateHeight; row++) {
		for (unsigned col = 0; col < partWidthWords; col++) {
			idx = row*widthRows + w0row + col;
			word1 = rotatedTemplate[idx];
			mask1 = rotatedMask[idx];
			if (threadIdx.x == 0) {
				words2[part] = otherTemplate[idx];
				masks2[part] = otherMask[idx];
			}
			__syncthreads();

			word2 = words2[part];
			mask2 = masks2[part];

			uint32_t x = XOR(word1, word2, mask1, mask2);
			nonZeroBits += __popc(x);
			totalBits += __popc(mask1 & mask2);
		}
	}

	hammingDistances[part][threadIdx.x] = float(nonZeroBits) / float(totalBits);
	__syncthreads();

	if (threadIdx.x == 0) {
		float minHD = 1.0;
		for (int i = 0; i < blockDim.x; i++) {
			minHD = min(minHD, hammingDistances[part][i]);
		}
		distances[nParts*templateIdx+part] = minHD;
	}
}

// Wrapper functions to invoke from C++
void doGPUMatchKernelWrapper(dim3 blockSize, dim3 gridSize, const uint8_t* d_rotatedTemplates, const uint8_t* d_rotatedMasks, size_t nRotatedTemplates, const GPUDatabase database, float* d_distances)
{
	doGPUMatchKernel<<<gridSize, blockSize>>>(
		d_rotatedTemplates,
		d_rotatedMasks,
		nRotatedTemplates,
		database,
		d_distances
	);
}

void doGPUAContrarioMatchKernelWrapper(dim3 blockSize, dim3 gridSize, const uint8_t* d_rotatedTemplates, const uint8_t* d_rotatedMasks, size_t nRotatedTemplates, const GPUDatabase database, float* d_distances)
{
	doGPUAContrarioMatchKernel<<<gridSize, blockSize>>>(
		d_rotatedTemplates,
		d_rotatedMasks,
		nRotatedTemplates,
		database,
		d_distances
	);
}
