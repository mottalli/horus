#include <vector>
#include <stdint.h>
#include <cassert>
#include <iostream>
#include <stdio.h>
#include "cudacommon.h"
#include "clock.h"


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

// Kernels (defined in irisdatabase_kernel.cu)
__global__ void doGPUMatchKernel(const uint8_t* rotatedTemplates, const uint8_t* rotatedMasks, size_t nRotatedTemplates, const GPUDatabase database, float* distances);
__global__ void doGPUAContrarioMatchKernel(const uint8_t* rotatedTemplates, const uint8_t* rotatedMasks, size_t nRotatedTemplates, const GPUDatabase database, float* distances);

namespace gpu {

	/**
	 * Load the database in the GPU
	 */
	void loadDatabase(const vector<IrisTemplate*>& templates, GPUDatabase& gpuDatabase)
	{
		cleanupDatabase(&gpuDatabase);

		size_t n = templates.size();

		if (n <= 0) {
			return;
		}

		const Mat& sampleTemplate = templates[0]->getPackedTemplate();
		size_t templateWidth = sampleTemplate.cols, templateHeight = sampleTemplate.rows;
		size_t templateSize = templateWidth*templateHeight;
		gpuDatabase.templateWidth = templateWidth;
		gpuDatabase.templateHeight = templateHeight;
		gpuDatabase.numberOfTemplates = n;

		size_t bytes = n*templateSize;
		CUDA_SAFE_CALL(cudaMalloc(&gpuDatabase.d_templates, bytes));
		CUDA_SAFE_CALL(cudaMalloc(&gpuDatabase.d_masks, bytes));

		for (size_t i = 0; i < n; i++) {
			const Mat& packedTemplate = templates[i]->getPackedTemplate();
			const Mat& packedMask = templates[i]->getPackedMask();

			assert(packedTemplate.isContinuous() && packedMask.isContinuous());
			assert(packedTemplate.channels() == 1 && packedMask.channels() == 1);
			assert(packedTemplate.type() == CV_8U && packedMask.type() == CV_8U);
			assert(packedTemplate.size() == packedMask.size());
			assert(packedTemplate.cols == gpuDatabase.templateWidth);
			assert(packedTemplate.rows == gpuDatabase.templateHeight);

			// Copy the template and mask to the GPU
			CUDA_SAFE_CALL(cudaMemcpy(gpuDatabase.d_templates + i*templateSize, packedTemplate.data, templateSize, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(gpuDatabase.d_masks + i*templateSize, packedMask.data, templateSize, cudaMemcpyHostToDevice));
		}
	};

	void doGPUMatch(const TemplateComparator& comparator, GPUDatabase& gpuDatabase, vector<double>& resultDistances, double& matchingTime)
	{
		Clock clock;
		clock.start();

		const std::vector<IrisTemplate>& rotatedTemplates = comparator.rotatedTemplates;
		size_t n = gpuDatabase.numberOfTemplates;

		assert(rotatedTemplates.size() < MAX_ROTS);

		// Load the rotated templates and masks to the GPU
		size_t templateSize = gpuDatabase.templateWidth * gpuDatabase.templateHeight;
		uint8_t *d_rotatedTemplates, *d_rotatedMasks;
		size_t bytes = rotatedTemplates.size() * templateSize;

		CUDA_SAFE_CALL(cudaMalloc(&d_rotatedTemplates, bytes));
		CUDA_SAFE_CALL(cudaMalloc(&d_rotatedMasks, bytes));
		for (size_t i = 0; i < rotatedTemplates.size(); i++) {
			const Mat& packedTemplate = rotatedTemplates[i].getPackedTemplate();
			const Mat& packedMask = rotatedTemplates[i].getPackedMask();

			assert(packedTemplate.isContinuous() && packedMask.isContinuous());
			assert(packedTemplate.channels() == 1 && packedMask.channels() == 1);
			assert(packedTemplate.type() == CV_8U && packedMask.type() == CV_8U);
			assert(packedTemplate.size() == packedMask.size());
			assert(packedTemplate.cols == gpuDatabase.templateWidth);
			assert(packedTemplate.rows == gpuDatabase.templateHeight);

			CUDA_SAFE_CALL(cudaMemcpy(d_rotatedTemplates + i*templateSize, packedTemplate.data, templateSize, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(d_rotatedMasks + i*templateSize, packedMask.data, templateSize, cudaMemcpyHostToDevice));
		}

		// Output buffer in device
		float* d_distances;
		CUDA_SAFE_CALL(cudaMalloc(&d_distances, n*sizeof(float)));


		// Invoke the kernel
		dim3 blockSize(rotatedTemplates.size(), 1, 1);
		dim3 gridSize(n, 1);

		doGPUMatchKernel<<<gridSize, blockSize>>>(
			d_rotatedTemplates,
			d_rotatedMasks,
			rotatedTemplates.size(),
			gpuDatabase,
			d_distances
		);

		// Retrieve the result
		float* distances = new float[n];
		cudaMemcpy(distances, d_distances, n*sizeof(float), cudaMemcpyDeviceToHost);


		// Copy the results (cast to double)
		for (size_t i = 0; i < n; i++) {
			resultDistances[i] = double(distances[i]);
		}

		// Free the memory
		CUDA_SAFE_CALL(cudaFree(d_rotatedTemplates));
		CUDA_SAFE_CALL(cudaFree(d_rotatedMasks));
		CUDA_SAFE_CALL(cudaFree(d_distances));
		free(distances);

		matchingTime = clock.stop();
	};

	void doGPUAContrarioMatch(const TemplateComparator& comparator, GPUDatabase& gpuDatabase, unsigned nParts, vector< vector<double> >& resultDistances, double& matchingTime)
	{
		assert(resultDistances.size() == nParts);

		const std::vector<IrisTemplate>& rotatedTemplates = comparator.rotatedTemplates;
		size_t n = gpuDatabase.numberOfTemplates;

		assert(rotatedTemplates.size() < MAX_ROTS);

		Clock clock;
		clock.start();

		// Load the rotated templates and masks to the GPU
		size_t templateSize = gpuDatabase.templateWidth * gpuDatabase.templateHeight;
		uint8_t *d_rotatedTemplates, *d_rotatedMasks;
		size_t bytes = rotatedTemplates.size() * templateSize;

		CUDA_SAFE_CALL(cudaMalloc(&d_rotatedTemplates, bytes));
		CUDA_SAFE_CALL(cudaMalloc(&d_rotatedMasks, bytes));
		for (size_t i = 0; i < rotatedTemplates.size(); i++) {
			const Mat& packedTemplate = rotatedTemplates[i].getPackedTemplate();
			const Mat& packedMask = rotatedTemplates[i].getPackedMask();

			assert(packedTemplate.isContinuous() && packedMask.isContinuous());
			assert(packedTemplate.channels() == 1 && packedMask.channels() == 1);
			assert(packedTemplate.type() == CV_8U && packedMask.type() == CV_8U);
			assert(packedTemplate.size() == packedMask.size());
			assert(packedTemplate.cols == gpuDatabase.templateWidth);
			assert(packedTemplate.rows == gpuDatabase.templateHeight);

			CUDA_SAFE_CALL(cudaMemcpy(d_rotatedTemplates + i*templateSize, packedTemplate.data, templateSize, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(d_rotatedMasks + i*templateSize, packedMask.data, templateSize, cudaMemcpyHostToDevice));
		}


		// Output buffer in device
		float* d_distances;
		CUDA_SAFE_CALL(cudaMalloc(&d_distances, n*nParts*sizeof(float)));


		// Invoke the kernel
		dim3 blockSize(rotatedTemplates.size(), nParts, 1);
		dim3 gridSize(n, 1);

		doGPUAContrarioMatchKernel<<<gridSize, blockSize>>>(
			d_rotatedTemplates,
			d_rotatedMasks,
			rotatedTemplates.size(),
			gpuDatabase,
			d_distances
		);

		// Retrieve the result
		float* distances = new float[n*nParts];
		cudaMemcpy(distances, d_distances, n*sizeof(float)*nParts, cudaMemcpyDeviceToHost);


		// Copy the results
		for (size_t i = 0; i < n; i++) {
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
}
