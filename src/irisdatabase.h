#pragma once
#include <vector>
#include <map>

#include "iristemplate.h"
#include "clock.h"

namespace horus {

typedef pair<int, double> MatchDistance;

class IrisDatabase
{
public:
	IrisDatabase();
	virtual ~IrisDatabase();

	virtual void addTemplate(int templateId, const IrisTemplate& irisTemplate);
	virtual void deleteTemplate(int templateId);
	virtual void doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int) = NULL, int nRots=20, int rotStep=2);
	virtual void doAContrarioMatch(const IrisTemplate& irisTemplate, int nParts=4, void (*statusCallback)(int) = NULL, int nRots=20, int rotStep=2);

	inline size_t databaseSize() const { return this->templates.size(); }

	inline int getMinDistanceId() const { return this->matchingDistances[0].first; }
	inline double getMinDistance() const { return this->matchingDistances[0].second; }
	inline double getDistanceFor(int templateId) { return this->distances[this->positions[templateId]]; }
	inline const vector<MatchDistance>& getMatchingDistances() const { return this->matchingDistances; }
	inline const vector<double>& getDistances() const { return this->distances; }
	inline const GrayscaleImage& getComparationImage() const { return this->comparationImage; }

	inline int getMinNFAId() const { return this->minNFAId; }
	inline double getMinNFA() const { return this->minNFA; }
	inline double getNFAFor(int templateId) { return this->resultNFAs[this->positions[templateId]]; }

	double getMatchingTime() const { return this->matchingTime; }

	//TODO: Move these as protected members
	vector<int> ids;
	vector<double> resultNFAs;
	vector< vector<double> > resultPartsDistances;

protected:
	virtual void calculatePartsDistances(const IrisTemplate& irisTemplate, unsigned int nParts, unsigned int nRots, unsigned int rotStep);

	vector<IrisTemplate> templates;
	map<int, int> positions;

	vector<MatchDistance> matchingDistances;
	vector<double> distances;
	static inline bool matchingDistanceComparator(MatchDistance d1, MatchDistance d2) { return d1.second < d2.second; }

	Timer timer;
	double matchingTime;

	double minNFA;
	int minNFAId;	

	GrayscaleImage comparationImage;

	int ignoreId;
};

}
