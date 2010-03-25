#include <vector>
#include <stdint.h>
#include <cassert>
#include <iostream>
#include "cudacommon.h"
#include "clock.h"

using namespace std;

// Taken from the CUDA SDK
#define CUDA_SAFE_CALL(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        fprintf(stderr, "CUDA_SAFE_CALL() Runtime API error in file <%s>, line %i : %s.\n",
                file, line, cudaGetErrorString( err) );
        exit(-1);
    }
}


#define XOR(a, b, mask1, mask2) ((a ^ b) & mask1 & mask2)
#define MAX_ROTS 40
#define MAX_PARTS 8

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

	assert((database.templateWidth % 4) == 0);

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

/**
 * Load the database in the GPU
 */
void loadDatabase(const vector<const uint8_t*>& templates, const vector<const uint8_t*>& masks, size_t templateWidth, size_t templateHeight, GPUDatabase* database)
{
	assert(templateWidth % 4 == 0);			// For casting to int32 in the GPU (4x speedup)
	size_t templateSize = templateWidth*templateHeight;

	cleanupDatabase(database);

	database->templateWidth = templateWidth;
	database->templateHeight = templateHeight;
	database->numberOfTemplates = templates.size();

	size_t bytes = templates.size()*templateSize;
	CUDA_SAFE_CALL(cudaMalloc(&database->d_templates, bytes));
	CUDA_SAFE_CALL(cudaMalloc(&database->d_masks, bytes));

	// Load each individual template in a contiguous chunk of GPU memory
	for (size_t i = 0; i < templates.size(); i++) {
		CUDA_SAFE_CALL(cudaMemcpy(database->d_templates + i*templateSize, templates[i], templateSize, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(database->d_masks + i*templateSize, masks[i], templateSize, cudaMemcpyHostToDevice));
	}
};

void doGPUMatch(const vector<const uint8_t*>& rotatedTemplates, const vector<const uint8_t*>& rotatedMasks, GPUDatabase* database, vector<double>& resultDistances, double& matchingTime)
{
	assert(rotatedTemplates.size() == rotatedMasks.size());
	assert(rotatedTemplates.size() < MAX_ROTS);
	assert(resultDistances.size() == database->numberOfTemplates);

	Clock clock;
	clock.start();

	// Load the rotated templates and masks to the GPU
	uint8_t *d_rotatedTemplates, *d_rotatedMasks;
	size_t templateSize = database->templateWidth * database->templateHeight;
	size_t bytes = rotatedTemplates.size() * templateSize;
	
	CUDA_SAFE_CALL(cudaMalloc(&d_rotatedTemplates, bytes));
	CUDA_SAFE_CALL(cudaMalloc(&d_rotatedMasks, bytes));
	for (size_t i = 0; i < rotatedTemplates.size(); i++) {
		CUDA_SAFE_CALL(cudaMemcpy(d_rotatedTemplates + i*templateSize, rotatedTemplates[i], templateSize, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_rotatedMasks + i*templateSize, rotatedMasks[i], templateSize, cudaMemcpyHostToDevice));
	}

	// Output buffer in device
	float* d_distances;
	CUDA_SAFE_CALL(cudaMalloc(&d_distances, database->numberOfTemplates*sizeof(float)));


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


	// Copy the results
	for (size_t i = 0; i < database->numberOfTemplates; i++) {
		resultDistances[i] = double(distances[i]);
	}

	// Free the memory
	CUDA_SAFE_CALL(cudaFree(d_rotatedTemplates));
	CUDA_SAFE_CALL(cudaFree(d_rotatedMasks));
	CUDA_SAFE_CALL(cudaFree(d_distances));
	free(distances);

	matchingTime = clock.stop();
};

void doGPUAContrarioMatch(const vector<const uint8_t*>& rotatedTemplates, const vector<const uint8_t*>& rotatedMasks, GPUDatabase* database, unsigned nParts, vector< vector<double> >& resultDistances, double& matchingTime)
{
	assert(rotatedTemplates.size() == rotatedMasks.size());
	assert(rotatedTemplates.size() < MAX_ROTS);
	assert(resultDistances.size() == database->numberOfTemplates);

	Clock clock;
	clock.start();

	// Load the rotated templates and masks to the GPU
	uint8_t *d_rotatedTemplates, *d_rotatedMasks;
	size_t templateSize = database->templateWidth * database->templateHeight;
	size_t bytes = rotatedTemplates.size() * templateSize;

	CUDA_SAFE_CALL(cudaMalloc(&d_rotatedTemplates, bytes));
	CUDA_SAFE_CALL(cudaMalloc(&d_rotatedMasks, bytes));
	for (size_t i = 0; i < rotatedTemplates.size(); i++) {
		CUDA_SAFE_CALL(cudaMemcpy(d_rotatedTemplates + i*templateSize, rotatedTemplates[i], templateSize, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_rotatedMasks + i*templateSize, rotatedMasks[i], templateSize, cudaMemcpyHostToDevice));
	}

	// Output buffer in device
	float* d_distances;
	CUDA_SAFE_CALL(cudaMalloc(&d_distances, database->numberOfTemplates*nParts*sizeof(float)));


	// Invoke the kernel
	dim3 blockSize(rotatedTemplates.size(), nParts, 1);
	dim3 gridSize(database->numberOfTemplates, 1);

	doGPUAContrarioMatchKernel<<<gridSize, blockSize>>>(
		d_rotatedTemplates,
		d_rotatedMasks,
		rotatedTemplates.size(),
		*database,
		d_distances
	);

	// Retrieve the result
	float* distances = new float[database->numberOfTemplates*nParts];
	cudaMemcpy(distances, d_distances, database->numberOfTemplates*sizeof(float)*nParts, cudaMemcpyDeviceToHost);


	// Copy the results
	for (size_t i = 0; i < database->numberOfTemplates; i++) {
		for (size_t p = 0; p < nParts; p++) {
			resultDistances[p][i] = double(distances[i*nParts+p]);
		}
	}

	// Free the memory
	CUDA_SAFE_CALL(cudaFree(d_rotatedTemplates));
	CUDA_SAFE_CALL(cudaFree(d_rotatedMasks));
	CUDA_SAFE_CALL(cudaFree(d_distances));
	free(distances);

	matchingTime = clock.stop();
};

void cleanupDatabase(GPUDatabase* database)
{
	if (database->d_templates != NULL) {
		cudaFree(database->d_templates);
		cudaFree(database->d_masks);
	}
}
