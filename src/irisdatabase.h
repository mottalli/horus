#pragma once
#include <vector>
#include <map>

#include "iristemplate.h"
#include "clock.h"

using namespace std;

typedef pair<int, double> MatchDistance;

class IrisDatabase
{
public:
	IrisDatabase();
	virtual ~IrisDatabase();

	void addTemplate(int templateId, const IrisTemplate& irisTemplate);
	void deleteTemplate(int templateId);

	void doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int) = NULL, int nRots=20, int rotStep=2);
	void doAContrarioMatch(const IrisTemplate& irisTemplate, int nParts=4, void (*statusCallback)(int) = NULL, int nRots=20, int rotStep=2);
	inline size_t databaseSize() const { return this->templates.size(); }

	inline int getMinDistanceId() const { return this->minDistanceId; }
	inline double getMinDistance() const { return this->minDistance; }
	inline double getDistanceFor(int templateId) { return this->distances[this->positions[templateId]]; }
	inline const vector<MatchDistance>& getMatchingDistances() const { return this->matchingDistances; }

	inline int getMinNFAId() const { return this->minNFAId; }
	inline double getMinNFA() const { return this->minNFA; }
	inline double getNFAFor(int templateId) { return this->resultNFAs[this->positions[templateId]]; }


	double getMatchingTime() const { return this->matchingTime; }

	vector<int> ids;

	vector< vector<double> > resultPartsDistances;
	vector<double> resultNFAs;

	GrayscaleImage comparationImage;

	int ignoreId;

protected:
	virtual void calculatePartsDistances(const IrisTemplate& irisTemplate, unsigned int nParts, unsigned int nRots, unsigned int rotStep);

	vector<IrisTemplate*> templates;
	map<int, int> positions;

	vector<MatchDistance> matchingDistances;
	vector<double> distances;
	static bool matchingDistanceComparator(MatchDistance d1, MatchDistance d2) { return d1.first < d2.first; }

	int minDistanceId;
	double minDistance;
	Clock clock;
	double matchingTime;

	double minNFA;
	int minNFAId;


};

