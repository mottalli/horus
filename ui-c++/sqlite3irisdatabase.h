#ifndef SQLITE3IRISDATABASE_H
#define SQLITE3IRISDATABASE_H

#include <iostream>
#include <QDebug>
#include <QObject>

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


	IrisData getIrisData(int userId) const;
	void addUser(string userName, const IrisTemplate& irisTemplate, const SegmentationResult& segmentationResult, const Mat image=Mat());
	void addImage(int userId, const Mat& image);

private:
	string dbPath;
	mutable sqlite3* db;

	void VERIFY_SQL(int status, const string msgError = "") const;
};

#endif // SQLITE3IRISDATABASE_H
