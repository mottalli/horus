#ifndef SQLITE3IRISDATABASE_H
#define SQLITE3IRISDATABASE_H

#include <iostream>
#include <QDebug>

#include "common.h"
#include "external/sqlite3/sqlite3.h"

class SQLite3IrisDatabase : public IrisDatabase
{
public:
	SQLite3IrisDatabase(const string& dbPath);
	~SQLite3IrisDatabase();

	typedef struct {
		int userId;
		string userName;
		IrisTemplate irisTemplate;
		SegmentationResult segmentation;
		Mat image;
	} IrisData;


	const IrisData getIrisData(int userId);

private:
	string dbPath;
	sqlite3* db;

	void VERIFY_SQL(int status, const string msgError = "");
};

#endif // SQLITE3IRISDATABASE_H
