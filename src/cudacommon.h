#pragma once

#include <stdint.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

const unsigned MAX_ROTS=40;
const unsigned MAX_PARTS=8;

// Taken from the CUDA SDK
#define CUDA_SAFE_CALL(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError err, const char *file, const int line )
{
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA_SAFE_CALL() Runtime API error in file <%s>, line %i : %s.\n",
				file, line, cudaGetErrorString(err));
		//TODO: throw exception instead
		exit(-1);
	}
}

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

// Wrapper functions (must be defined as "extern C")
extern "C" {
	void doGPUMatchKernelWrapper(dim3 blockSize, dim3 gridSize, const uint8_t* rotatedTemplates, const uint8_t* rotatedMasks, size_t nRotatedTemplates, const GPUDatabase database, float* distances);
	void doGPUAContrarioMatchKernelWrapper(dim3 blockSize, dim3 gridSize, const uint8_t* rotatedTemplates, const uint8_t* rotatedMasks, size_t nRotatedTemplates, const GPUDatabase database, float* distances);
}
