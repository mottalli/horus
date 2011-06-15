#include "irisdatabasecuda.h"
#include "templatecomparator.h"

using namespace horus;
using namespace cv;

IrisDatabaseCUDA::IrisDatabaseCUDA()
{
	this->dirty = true;
}

IrisDatabaseCUDA::~IrisDatabaseCUDA()
{
	this->cleanupDB();
}

void IrisDatabaseCUDA::addTemplate(int templateId, const IrisTemplate& irisTemplate)
{
	IrisDatabase::addTemplate(templateId, irisTemplate);
	this->dirty = true;
}

void IrisDatabaseCUDA::deleteTemplate(int templateId)
{
	IrisDatabase::deleteTemplate(templateId);
	this->dirty = true;
}

void IrisDatabaseCUDA::calculatePartsDistances(const IrisTemplate& irisTemplate, int nParts, int nRots, int rotStep)
{
	size_t n = this->templates.size();

	assert(this->resultPartsDistances.size() == nParts);
	assert(this->resultPartsDistances[0].size() == n);

	if (this->dirty) {
		this->uploadDBToDevice();
		this->dirty = false;
	}
	
	// Load the rotated templates and masks to the GPU
	TemplateComparator comparator(irisTemplate, nRots, rotStep);
	const vector<IrisTemplate>& rotatedTemplates = comparator.rotatedTemplates;
	std::pair<uint8_t*, uint8_t*> d_addresses = this->uploadRotatedTemplates(rotatedTemplates);
	uint8_t* d_rotatedTemplates = d_addresses.first;
	uint8_t* d_rotatedMasks = d_addresses.second;

	// Output buffer in device
	float* d_distances;
	CUDA_SAFE_CALL( cudaMalloc(&d_distances, n*nParts*sizeof(float)) );

	// Invoke the kernel
	dim3 blockSize(rotatedTemplates.size(), nParts, 1);
	dim3 gridSize(n, 1);

	::doGPUAContrarioMatchKernelWrapper(blockSize, gridSize, d_rotatedTemplates, d_rotatedMasks, rotatedTemplates.size(), gpuDatabase, d_distances);

	// Retrieve the result
	float* distances = new float[n*nParts];
	cudaMemcpy(distances, d_distances, n*sizeof(float)*nParts, cudaMemcpyDeviceToHost);

	// Copy the results
	for (size_t i = 0; i < n; i++) {
		for (size_t p = 0; p < nParts; p++) {
			this->resultPartsDistances[p][i] = double(distances[i*nParts+p]);
		}
	}

	// Free the memory
	CUDA_SAFE_CALL( cudaFree(d_rotatedTemplates) );
	CUDA_SAFE_CALL( cudaFree(d_rotatedMasks) );
	CUDA_SAFE_CALL( cudaFree(d_distances) );
	free(distances);
}

void IrisDatabaseCUDA::doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int), int nRots, int rotStep)
{
	this->timer.restart();

	size_t n = this->templates.size();

	if (this->dirty) {
		this->uploadDBToDevice();
		this->dirty = false;

		this->distances = vector<double>(this->templates.size());
	}
	
	// Load the rotated templates and masks to the GPU
	TemplateComparator comparator(irisTemplate, nRots, rotStep);
	const vector<IrisTemplate>& rotatedTemplates = comparator.rotatedTemplates;
	std::pair<uint8_t*, uint8_t*> d_addresses = this->uploadRotatedTemplates(rotatedTemplates);
	uint8_t* d_rotatedTemplates = d_addresses.first;
	uint8_t* d_rotatedMasks = d_addresses.second;

	// Output buffer in device
	float* d_distances;
	CUDA_SAFE_CALL( cudaMalloc(&d_distances, n*sizeof(float)) );

	// Invoke the kernel
	dim3 blockSize(rotatedTemplates.size(), 1, 1);
	dim3 gridSize(n, 1);
	::doGPUMatchKernelWrapper(blockSize, gridSize, d_rotatedTemplates, d_rotatedMasks, rotatedTemplates.size(), gpuDatabase, d_distances);

	// Retrieve the results
	float* distances = new float[n];
	cudaMemcpy(distances, d_distances, n*sizeof(float), cudaMemcpyDeviceToHost);

	// Copy the results (cast to double)
	for (size_t i = 0; i < n; i++) {
		this->distances[i] = double(distances[i]);
	}

	// Free the memory
	CUDA_SAFE_CALL( cudaFree(d_rotatedTemplates) );
	CUDA_SAFE_CALL( cudaFree(d_rotatedMasks) );
	CUDA_SAFE_CALL( cudaFree(d_distances) );
	free(distances);

	this->matchingTime = this->timer.elapsed();
}

void IrisDatabaseCUDA::uploadDBToDevice()
{
	this->cleanupDB();			// Cleanup any previously uploaded data

	size_t n = this->templates.size();
	this->gpuDatabase.numberOfTemplates = n;
	if (n == 0) return;

	Size templateSize = this->templates[0].getPackedTemplate().size();
	this->gpuDatabase.templateWidth = templateSize.width;
	this->gpuDatabase.templateHeight = templateSize.height;

	size_t templateLength = templateSize.width*templateSize.height;
	size_t totalBytes = n*templateLength;
	// Reserve 2*totalBytes in the GPU for storing the templates and the masks as a continuous chunk of memory
	CUDA_SAFE_CALL( cudaMalloc(&this->gpuDatabase.d_templates, totalBytes) );
	CUDA_SAFE_CALL( cudaMalloc(&this->gpuDatabase.d_masks, totalBytes) );

	// Upload the templates and the masks to the GPU
	for (size_t i = 0; i < n; i++) {
		const Mat1b& packedTemplate = this->templates[i].getPackedTemplate();
		const Mat1b& packedMask = this->templates[i].getPackedMask();

		assert(packedTemplate.isContinous() && packedMask.isContinuous());			// Must be continuous to use cudaMemcpy

		uint8_t* ptrTemplateDevice = this->gpuDatabase.d_templates + i*templateLength;
		uint8_t* ptrMaskDevice = this->gpuDatabase.d_masks + i*templateLength;
		CUDA_SAFE_CALL( cudaMemcpy(ptrTemplateDevice, packedTemplate.data, templateLength, cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpy(ptrMaskDevice, packedMask.data, templateLength, cudaMemcpyHostToDevice) );
	}
}

void IrisDatabaseCUDA::cleanupDB()
{
	if (this->gpuDatabase.d_templates != NULL) {
		cudaFree(this->gpuDatabase.d_templates);
		cudaFree(this->gpuDatabase.d_masks);
	}
}

pair<uint8_t*, uint8_t*> IrisDatabaseCUDA::uploadRotatedTemplates(const vector<IrisTemplate>& rotatedTemplates)
{
	assert(rotatedTemplates.size() < MAX_ROTS);

	size_t templateSize = this->gpuDatabase.templateWidth * this->gpuDatabase.templateHeight;
	size_t bytes = rotatedTemplates.size() * templateSize;
	uint8_t *d_rotatedTemplates, *d_rotatedMasks;
	CUDA_SAFE_CALL( cudaMalloc(&d_rotatedTemplates, bytes) );
	CUDA_SAFE_CALL( cudaMalloc(&d_rotatedMasks, bytes) );
	for (size_t i = 0; i < rotatedTemplates.size(); i++) {
		const Mat1b& packedTemplate = rotatedTemplates[i].getPackedTemplate();
		const Mat1b& packedMask = rotatedTemplates[i].getPackedMask();

		assert(packedTemplate.isContinuous() && packedMask.isContinuous());

		uint8_t* d_rotatedTemplate = d_rotatedTemplates + i*templateSize;
		uint8_t* d_rotatedMask = d_rotatedMasks + i*templateSize;

		CUDA_SAFE_CALL(cudaMemcpy(d_rotatedTemplate, packedTemplate.data, templateSize, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_rotatedMask, packedMask.data, templateSize, cudaMemcpyHostToDevice));
	}

	return pair<uint8_t*,uint8_t*>(d_rotatedTemplates, d_rotatedMasks);
}

void IrisDatabaseCUDA::doAContrarioMatch(const IrisTemplate& irisTemplate, int nParts, void (*)(int), int nRots, int rotStep)
{
	this->timer.restart();
	unsigned const int BINS = this->templates.size()/2;
	assert(BINS >= 1);
	const float BIN_MIN = 0.0f;
	const float BIN_MAX = 0.7f;

	IplImage** distances = new IplImage*[nParts];			// Should be CvMat but OpenCV doesn't let you use CvMat for calculating histograms
	size_t n = this->templates.size();
	this->resultPartsDistances = vector< vector<double> >(nParts, vector<double>(n));		// This is a copy in a better format to interface with Python

	this->calculatePartsDistances(irisTemplate, nParts, nRots, rotStep);

	for (int p = 0; p < nParts; p++) {
		//distances[p] = cvCreateMat(1, n, CV_32F);
		distances[p] = cvCreateImage(cvSize(1, n), IPL_DEPTH_32F, 1);
		for (size_t i = 0; i < n; i++) {
			cvSetReal1D(distances[p], i, this->resultPartsDistances[p][i]);
		}
	}

	// Calculate the histogram for the distances of each part
	float l_range[] = {BIN_MIN, BIN_MAX};		// The histogram has BINS bins equally distributed between BIN_MIN and BIN_MAX
	float* range[] = { l_range };
	int size[] = { BINS };

	CvHistogram** histograms = new CvHistogram*[nParts];

	for (int p = 0; p < nParts; p++) {
		histograms[p] = cvCreateHist(1, size, CV_HIST_ARRAY, range, 1);
		IplImage* a[] = { distances[p] };
		cvCalcHist(a, histograms[p]);
	}

	// Calculate the cumulative of the histograms
	float** cumhists = new float*[nParts];
	for (int p = 0; p < nParts; p++) {
		float* cumhist = (float*)malloc(BINS*sizeof(float));

		cumhist[0] = cvQueryHistValue_1D(histograms[p], 0);
		for (unsigned int i = 1; i < BINS; i++) {
			cumhist[i] = cumhist[i-1] + cvQueryHistValue_1D(histograms[p], i);
		}

		cumhists[p] = cumhist;
	}


	// Now calculate the NFA between the template and all the templates in the database
	this->resultNFAs = vector<double>(n);
	this->minNFA = INT_MAX;


	size_t bestIdx = 0;
	for (unsigned int i = 0; i < n; i++) {
		this->resultNFAs[i] = log10(double(n));

		for (int p = 0; p < nParts; p++) {
			double distance = cvGetReal1D(distances[p], i);
			unsigned int bin = floor( distance / ((BIN_MAX-BIN_MIN)/BINS) );
			assert(bin < BINS);

			this->resultNFAs[i] += log10( double(cumhists[p][bin]) / double(n) );		// The accumulated histogram has to be normalized, so we divide by n
		}

		int matchId = this->ids[i];

		if (matchId != this->ignoreId && this->resultNFAs[i] < this->minNFA) {
			this->minNFA = this->resultNFAs[i];
			this->minNFAId = matchId;
			bestIdx = i;
		}
	}

	// Clean up
	for (int p = 0; p < nParts; p++) {
		cvReleaseHist(&histograms[p]);
		cvReleaseImage(&distances[p]);
		free(cumhists[p]);
	}

	delete[] distances;
	delete[] histograms;
	delete[] cumhists;

	this->matchingTime = this->timer.elapsed();

	// Generate the comparation image
	TemplateComparator comparator(irisTemplate, nRots, rotStep);
	comparator.compare(this->templates[bestIdx]);
	this->comparationImage = comparator.getComparationImage();
}
