#include <algorithm>
#ifdef HORUS_MULTITHREAD_SEARCH
#include <boost/thread.hpp>
#endif

#include "common.h"
#include "irisdatabase.h"
#include "templatecomparator.h"
#include "tools.h"

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

	// Thread code for parallel DB matching
	struct {
		void operator() (size_t i0, size_t i1, const TemplateComparator& comparator, IrisDatabase* db) {
			for (size_t i = i0; i < i1; i++) {
				double hammingDistance = comparator.compare(db->templates[i]);
				int matchId = db->ids[i];
				db->matchingDistances[i] = MatchDistance(matchId, hammingDistance);
				db->distances[i] = hammingDistance;
			}
		}
	} matchingThread;

#ifdef HORUS_MULTITHREAD_SEARCH
	// Do a multi-threaded search
	boost::thread_group tg;
	const size_t nThreads = boost::thread::hardware_concurrency();			// Number of threads
	size_t step = (n/nThreads) + 1;											// Number of items to process per thread

	for (size_t i = 0; i < nThreads; i++) {
		size_t i0 = i*step;
		size_t i1 = min((i+1)*step, n);

		// Start up the matching thread
		tg.create_thread( boost::bind<void>(matchingThread, i0, i1, boost::ref(comparator), this) );

		// If i == nThreads-1 => i1 == n (on the last thread, it must process up to the last element)
		assert(i != nThreads-1 || i1 == n);
	}
	tg.join_all();
#else
	// Do a single-threaded search in the current thread
	matchingThread(0, n, comparator, this);
#endif

	this->matchingTime = this->timer.elapsed();

	// Sort the results from minimum to maximum distance
	sort(this->matchingDistances.begin(), this->matchingDistances.end(), IrisDatabase::matchingDistanceComparator);
	if (statusCallback) statusCallback(100);

	const IrisTemplate& matchingTemplate = this->templates[ this->positions[this->getMinDistanceId()] ];
	this->comparationImage = TemplateComparator::getComparationImage(matchingTemplate, true);
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

void IrisDatabase::doAContrarioMatch(const IrisTemplate& irisTemplate, size_t nParts, void (*)(int), int nRots, int rotStep)
{
	this->timer.restart();

	unsigned const int BINS = this->templates.size()/2;
	assert(BINS >= 1);
	size_t n = this->templates.size();

	this->resultPartsDistances = vector< vector<double> >(nParts, vector<double>(n));		// This is a copy in a better format to interface with Python
	this->calculatePartsDistances(irisTemplate, nParts, nRots, rotStep);

	tools::Histogram* cumhists = new tools::Histogram[nParts];
	for (size_t p = 0; p < nParts; p++) {
		cumhists[p] = tools::Histogram(this->resultPartsDistances[p], BINS).cumulative();
	}

	// Now calculate the NFA between the template and all the templates in the database
	this->resultNFAs = vector<double>(n);
	this->minNFA = INT_MAX;

	size_t bestIdx = 0;
	for (unsigned int i = 0; i < n; i++) {
		this->resultNFAs[i] = log10(double(n));

		for (size_t p = 0; p < nParts; p++) {
			double distance = this->resultPartsDistances[p][i];
			size_t bin = cumhists[p].binFor(distance);
			double histval = cumhists[p].values[bin];

			this->resultNFAs[i] += log10(histval);
		}

		int matchId = this->ids[i];

		if (matchId != this->ignoreId && this->resultNFAs[i] < this->minNFA) {
			this->minNFA = this->resultNFAs[i];
			this->minNFAId = matchId;
			bestIdx = i;
		}
	}

	delete[] cumhists;

	this->matchingTime = this->timer.elapsed();
}
