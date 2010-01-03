#include <algorithm>

#include "irisdatabase.h"
#include "templatecomparator.h"

IrisDatabase::IrisDatabase()
{
	this->ignoreId = 0;
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

	this->resultDistances.clear();
	this->resultDistances.reserve(this->templates.size());

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
