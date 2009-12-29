#ifndef IRISDATABASE_H
#define IRISDATABASE_H

#include <map>
#include "iristemplate.h"

using namespace std;

class IrisDatabase
{
public:
	IrisDatabase();
	~IrisDatabase();

	void addTemplate(int templateId, const IrisTemplate& irisTemplate);

protected:
	map<int, IrisTemplate*> database;
};

#endif // IRISDATABASE_H
