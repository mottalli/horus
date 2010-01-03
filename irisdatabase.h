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

	void doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int) = NULL, int nRots=2, int rotStep=2);

	inline int getMinDistanceId() const { return this->minDistanceId; };
	inline double getMinDistance() const { return this->minDistance; };

	inline double getMatchingTime() const { return this->clock.time(); };

	vector<int> ids;
	vector<float> resultDistances;

	int ignoreId;

protected:
	vector<IrisTemplate*> templates;

	int minDistanceId;
	double minDistance;
	Clock clock;
};

#endif // IRISDATABASE_H
