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

int loadCallback(void *NotUsed, int argc, char **argv, char **azColName);
int matchCallback(void *NotUsed, int argc, char **argv, char **azColName);

IrisDatabaseCUDA irisDatabaseCUDA;
IrisDatabase irisDatabase;


int main(int argc, char** argv)
{
	const char* pathBase = "/home/marcelo/iris/BBDD/base.db";
	sqlite3* db;
	char* errMsg;

	sqlite3_open(pathBase, &db);
	sqlite3_exec(db, "SELECT id_imagen,imagen,segmentacion,codigo_gabor FROM base_iris WHERE segmentacion_correcta=1", loadCallback, NULL, &errMsg);
	sqlite3_exec(db, "SELECT id_imagen,imagen,segmentacion,codigo_gabor FROM base_iris WHERE segmentacion_correcta=1", matchCallback, NULL, &errMsg);
	sqlite3_close(db);

	return 0;
}

int loadCallback(void *NotUsed, int argc, char **argv, char **azColName)
{
	string serializedTemplate = argv[3];
	int id_imagen = atoi(argv[0]);

	cout << "Cargando " << id_imagen << "..." << endl;

	IrisTemplate template_ = Serializer::unserializeIrisTemplate(serializedTemplate);
	irisDatabase.addTemplate(id_imagen, template_);
	irisDatabaseCUDA.addTemplate(id_imagen, template_);

	return 0;
}

int matchCallback(void *NotUsed, int argc, char **argv, char **azColName)
{
	string serializedTemplate = argv[3];
	int id_imagen = atoi(argv[0]);
	IrisTemplate template_ = Serializer::unserializeIrisTemplate(serializedTemplate);

	double tNormal, tCUDA;

	cout << "Matching normal " << id_imagen << "... ";
	irisDatabase.doMatch(template_);
	tNormal = irisDatabase.getMatchingTime();
	cout << tNormal << "ms" << endl;

	cout << "Matching CUDA " << id_imagen << "... ";
	irisDatabaseCUDA.doMatch(template_);
	tCUDA = irisDatabaseCUDA.getMatchingTime();
	cout << tCUDA << "ms" << endl;

	cout << "Mejora: " << tNormal/tCUDA << "x" << endl;
	
	return 0;
}
