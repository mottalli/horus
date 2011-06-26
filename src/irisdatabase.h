#pragma once
#include <boost/thread.hpp>

#include "common.h"
#include "iristemplate.h"
#include "clock.h"

namespace horus {

typedef std::pair<int, double> MatchDistance;

class IrisDatabase
{
public:
	IrisDatabase();
	virtual ~IrisDatabase();

	virtual void addTemplate(int templateId, const IrisTemplate& irisTemplate);
	virtual void deleteTemplate(int templateId);
	virtual void doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int) = NULL, int nRots=20, int rotStep=2);
	void doAContrarioMatch(const IrisTemplate& irisTemplate, size_t nParts=4, void (*statusCallback)(int) = NULL, int nRots=20, int rotStep=2);

	inline size_t databaseSize() const { return this->templates.size(); }

	inline int getMinDistanceId() const { return this->matchingDistances[0].first; }
	inline double getMinDistance() const { return this->matchingDistances[0].second; }
	inline double getDistanceFor(int templateId) { return this->distances[this->positions[templateId]]; }
	inline const std::vector<MatchDistance>& getMatchingDistances() const { return this->matchingDistances; }
	inline std::vector<double> getDistances() const { return this->distances; }
	inline const GrayscaleImage& getComparationImage() const { return this->comparationImage; }

	inline int getMinNFAId() const { return this->minNFAId; }
	inline double getMinNFA() const { return this->minNFA; }
	inline double getNFAFor(int templateId) { return this->resultNFAs[this->positions[templateId]]; }

	double getMatchingTime() const { return this->matchingTime; }

	//TODO: Move these as protected members
	std::vector<int> ids;
	std::vector<double> resultNFAs;
	std::vector< std::vector<double> > resultPartsDistances;

protected:
	virtual void calculatePartsDistances(const IrisTemplate& irisTemplate, unsigned int nParts, unsigned int nRots, unsigned int rotStep);

	std::vector<IrisTemplate> templates;
	std::map<int, int> positions;			// Maps an ID with the position in the template database

	std::vector<MatchDistance> matchingDistances;
	std::vector<double> distances;
	static inline bool matchingDistanceComparator(MatchDistance d1, MatchDistance d2) { return d1.second < d2.second; }

	Timer timer;
	double matchingTime;

	double minNFA;
	int minNFAId;	

	GrayscaleImage comparationImage;

	int ignoreId;
};

}
