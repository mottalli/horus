#include <algorithm>

#include "irisdatabase.h"
#include "templatecomparator.h"

using namespace horus;

IrisDatabase::IrisDatabase()
{
	this->ignoreId = -1;
}

IrisDatabase::~IrisDatabase()
{
}

void IrisDatabase::addTemplate(int templateId, const IrisTemplate& irisTemplate)
{
	if (find(this->ids.begin(), this->ids.end(), templateId) != this->ids.end()) {
		// The template already exists -- delete it
		this->deleteTemplate(templateId);
	}

	this->templates.push_back(irisTemplate);			// Note that this creates a copy of the template
	this->ids.push_back(templateId);
	this->positions[templateId] = this->ids.size()-1;
}

void IrisDatabase::deleteTemplate(int templateId)
{
	vector<int>::iterator it1;
	vector<IrisTemplate>::iterator it2;

	for (it1 = this->ids.begin(), it2 = this->templates.begin(); it1 != this->ids.end(); it1++, it2++) {
		if (*it1 == templateId) {
			this->ids.erase(it1);
			this->templates.erase(it2);
			this->positions.erase(templateId);
			break;
		}
	}
}

void IrisDatabase::doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int), int nRots, int rotStep)
{
	this->timer.restart();
	TemplateComparator comparator(irisTemplate, nRots, rotStep);

	size_t n = this->templates.size();

	this->matchingDistances = vector<MatchDistance>(n);
	this->distances = vector<double>(n);

	for (size_t i = 0; i < n; i++) {
		double hammingDistance = comparator.compare(this->templates[i]);
		int matchId = this->ids[i];
		this->matchingDistances[i] = MatchDistance(matchId, hammingDistance);
		this->distances[i] = hammingDistance;

		int percentage = (90*i)/n;			// The extra 10% is for sorting
		if (statusCallback && ((i % ((n/10)+1)) == 0)) statusCallback(percentage);
	}

	this->matchingTime = this->timer.elapsed();

	// Sort the results from minimum to maximum distance
	sort(this->matchingDistances.begin(), this->matchingDistances.end(), IrisDatabase::matchingDistanceComparator);

	//comparator.compare(*(this->templates[bestIdx]));
	//this->comparationImage = comparator.getComparationImage();
}

void IrisDatabase::calculatePartsDistances(const IrisTemplate& irisTemplate, unsigned int nParts, unsigned int nRots, unsigned int rotStep)
{
	size_t n = this->templates.size();

	assert(this->resultPartsDistances.size() == nParts);
	assert(this->resultPartsDistances[0].size() == n);

	TemplateComparator comparator(irisTemplate, nRots, rotStep);

	// Calculate the distances between the parts
	for (size_t i = 0; i < n; i++) {
		vector<double> partsDistances = comparator.compareParts(this->templates[i], nParts);
		assert(partsDistances.size() == nParts);

		for (unsigned int p = 0; p < nParts; p++) {
			this->resultPartsDistances[p][i] = partsDistances[p];
		}
	}
}

void IrisDatabase::doAContrarioMatch(const IrisTemplate& irisTemplate, int nParts, void (*)(int), int nRots, int rotStep)
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
