#include <iostream>
#include <stdio.h>
#include <sqlite3.h>

#include "common.h"
#include "segmentator.h"
#include "decorator.h"
#include "irisencoder.h"
#include "irisdctencoder.h"
#include "parameters.h"
#include "videoprocessor.h"
#include "templatecomparator.h"
#include "qualitychecker.h"
#include "tools.h"
#include "serializer.h"
#include "irisdatabasecuda.h"

using namespace std;

int callback(void *NotUsed, int argc, char **argv, char **azColName);

IrisTemplate templateToMatch;
bool gotMatch = false;

int main(int argc, char** argv)
{
	const char* pathBase = "/home/marcelo/iris/BBDD/Bath/base.db";
	sqlite3* db;
	char* errMsg;
	IrisDatabaseCUDA irisDatabase;

	sqlite3_open(pathBase, &db);
	sqlite3_exec(db, "SELECT id_imagen,imagen,segmentacion,codigo_gabor FROM base_iris WHERE segmentacion_correcta=1", callback, &irisDatabase, &errMsg);
	sqlite3_close(db);

	irisDatabase.doMatch(templateToMatch);

	return 0;
}

int callback(void *pDatabase, int argc, char **argv, char **azColName)
{
	IrisDatabaseCUDA* irisDatabase = (IrisDatabaseCUDA*)pDatabase;
	string serializedTemplate = argv[3];
	int id_imagen = atoi(argv[0]);

	cout << "Cargando " << id_imagen << "..." << endl;

	IrisTemplate template_ = Serializer::unserializeIrisTemplate(serializedTemplate);
	irisDatabase->addTemplate(id_imagen, template_);

	if (!gotMatch) {
		templateToMatch = template_;
		gotMatch = true;
	}

	return 0;
}
