#include <algorithm>

#include "irisdatabase.h"
#include "templatecomparator.h"

IrisDatabase::IrisDatabase()
{
	this->ignoreId = -1;
}

IrisDatabase::~IrisDatabase()
{
	for (vector<IrisTemplate*>::iterator it = this->templates.begin(); it != this->templates.end(); it++) {
		delete (*it);			// Free the memory allocated in addTemplate
	}
}

void IrisDatabase::addTemplate(int templateId, const IrisTemplate& irisTemplate)
{
	if (find(this->ids.begin(), this->ids.end(), templateId) != this->ids.end()) {
		// The template already exists -- delete it
		this->deleteTemplate(templateId);
	}

	IrisTemplate* newTemplate = new IrisTemplate(irisTemplate);
	this->templates.push_back(newTemplate);
	this->ids.push_back(templateId);
}

void IrisDatabase::deleteTemplate(int templateId)
{
	vector<int>::iterator it1;
	vector<IrisTemplate*>::iterator it2;

	for (it1 = this->ids.begin(), it2 = this->templates.begin(); it1 != this->ids.end(); it1++, it2++) {
		if (*it1 == templateId) {
			this->ids.erase(it1);
			this->templates.erase(it2);
			break;
		}
	}
}

void IrisDatabase::doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int), int nRots, int rotStep)
{
	this->clock.start();
	TemplateComparator comparator(irisTemplate, nRots, rotStep);

	size_t n = this->templates.size();
	this->minDistanceId = 0;
	this->minDistance = 1.0;

	this->resultDistances = vector<double>(n);

	for (size_t i = 0; i < n; i++) {
		double hammingDistance = comparator.compare(*(this->templates[i]));
		this->resultDistances[i] = hammingDistance;
		int matchId = this->ids[i];

		if (matchId != this->ignoreId && hammingDistance < this->minDistance) {
			this->minDistance = hammingDistance;
			this->minDistanceId = matchId;
		}

		int percentage = (100*i)/n;
		if (statusCallback) statusCallback(percentage);
	}

	this->clock.stop();
}

void IrisDatabase::doAContrarioMatch(const IrisTemplate& irisTemplate, int nParts, void (*statusCallback)(int), int nRots, int rotStep)
{
	this->clock.start();
	unsigned const int BINS = 70;
	const float BIN_MIN = 0.0f;
	const float BIN_MAX = 0.7f;

	IplImage* distances[nParts];			// Should be CvMat but OpenCV doesn't let you use CvMat for calculating histograms
	size_t n = this->templates.size();
	this->resultPartsDistances = vector< vector<double> >(nParts, vector<double>(n));		// This is a copy in a better format to interface with Python


	for (int p = 0; p < nParts; p++) {
		//distances[p] = cvCreateMat(1, n, CV_32F);
		distances[p] = cvCreateImage(cvSize(1, n), IPL_DEPTH_32F, 1);
	}

	TemplateComparator comparator(irisTemplate, nRots, rotStep);

	// Calculate the distances between the parts
	for (size_t i = 0; i < n; i++) {
		std::vector<double> partsDistances = comparator.compareParts(*(this->templates[i]), nParts);
		assert(partsDistances.size() == nParts);

		for (int p = 0; p < nParts; p++) {
			cvSetReal1D(distances[p], i, partsDistances[p]);
			this->resultPartsDistances[p][i] = partsDistances[p];
		}
	}

	// Calculate the histogram for the distances of each part
	float l_range[] = {BIN_MIN, BIN_MAX};		// The histogram has BINS bins equally distributed between BIN_MIN and BIN_MAX
	float* range[] = { l_range };
	int size[] = { BINS };

	CvHistogram* histograms[nParts];

	for (int p = 0; p < nParts; p++) {
		histograms[p] = cvCreateHist(1, size, CV_HIST_ARRAY, range, 1);
		IplImage* a[] = { distances[p] };
		cvCalcHist(a, histograms[p]);
	}

	// Calculate the cumulative of the histograms
	float* cumhists[nParts];
	for (int p = 0; p < nParts; p++) {
		float* cumhist = (float*)malloc(BINS*sizeof(float));

		cumhist[0] = cvQueryHistValue_1D(histograms[p], 0);
		for (int i = 1; i < BINS; i++) {
			cumhist[i] = cumhist[i-1] + cvQueryHistValue_1D(histograms[p], i);
		}

		cumhists[p] = cumhist;
	}
	

	// Now calculate the NFA between the template and all the templates in the database
	this->resultNFAs = vector<double>(n);
	this->minNFA = INT_MAX;

	for (int i = 0; i < n; i++) {
		double sum = 0.0;
		
		this->resultNFAs[i] = std::log10(double(n));

		for (int p = 0; p < nParts; p++) {
			double distance = cvGetReal1D(distances[p], i);
			int bin = std::floor( distance / ((BIN_MAX-BIN_MIN)/BINS) );
			assert(bin < BINS);

			this->resultNFAs[i] += std::log10( double(cumhists[p][bin]) / double(n) );		// The accumulated histogram has to be normalized, so we divide by n
		}

		int matchId = this->ids[i];

		if (matchId != this->ignoreId && this->resultNFAs[i] < this->minNFA) {
			this->minNFA = this->resultNFAs[i];
			this->minNFAId = matchId;
		}
	}

	for (int p = 0; p < nParts; p++) {
		cvReleaseHist(&histograms[p]);
		free(cumhists[p]);
	}

	this->clock.stop();
}
