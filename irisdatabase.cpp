#include "irisdatabase.h"

IrisDatabase::IrisDatabase()
{
}

IrisDatabase::~IrisDatabase()
{
	for (map<int, IrisTemplate*>::iterator it = this->database.begin(); it != this->database.end(); it++) {
		delete (*it).second;			// Free the memory allocated in addTemplate
	}
}

void IrisDatabase::addTemplate(int templateId, const IrisTemplate& irisTemplate)
{
	if (this->database.find(templateId) != this->database.end()) {
		// Already has the element
		delete this->database[templateId];
	}
	IrisTemplate* newTemplate = new IrisTemplate(irisTemplate);
	this->database[templateId] = newTemplate;
}
