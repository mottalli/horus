#ifndef IRISDATABASE_H
#define IRISDATABASE_H

#include <vector>

#include "iristemplate.h"
#include "clock.h"

using namespace std;

class IrisDatabase
{
public:
	IrisDatabase();
	~IrisDatabase();

	void addTemplate(int templateId, const IrisTemplate& irisTemplate);
	void deleteTemplate(int templateId);

	void doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int) = NULL, int nRots=20, int rotStep=2);
	void doAContrarioMatch(const IrisTemplate& irisTemplate, int nParts=4, void (*statusCallback)(int) = NULL, int nRots=20, int rotStep=2);

	inline int getMinDistanceId() const { return this->minDistanceId; };
	inline double getMinDistance() const { return this->minDistance; };

	inline int getMinNFAId() const { return this->minNFAId; };
	inline double getMinNFA() const { return this->minNFA; };

	inline double getMatchingTime() const { return this->clock.time(); };

	vector<int> ids;
	vector<float> resultDistances;

	int ignoreId;

protected:
	vector<IrisTemplate*> templates;

	int minDistanceId;
	double minDistance;
	Clock clock;

	double minNFA;
	int minNFAId;
};

#endif // IRISDATABASE_H
