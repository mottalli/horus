#ifndef SQLITE3IRISDATABASE_H
#define SQLITE3IRISDATABASE_H

#include <iostream>
#include <QDebug>
#include <QObject>
#include <boost/optional.hpp>

#include "common.h"
#include "external/sqlite3/sqlite3.h"

#ifdef HORUS_CUDA_SUPPORT
class SQLite3IrisDatabase : public IrisDatabaseCUDA
#else
class SQLite3IrisDatabase : public IrisDatabase
#endif
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
	void addUser(string userName, const IrisTemplate& irisTemplate, const SegmentationResult& segmentationResult, const Image& image);
	void addImage(int userId, const Image& image, const SegmentationResult& segmentationResult, optional<IrisTemplate> averageTemplate = optional<IrisTemplate>());

private:
	string dbPath;
	mutable sqlite3* db;

	inline string getImagePathForUser(int userId) const { return (boost::format("%s/%i.jpg") % this->dbPath % userId).str(); }

	void VERIFY_SQL(int status, const string msgError = "") const;
};

#endif // SQLITE3IRISDATABASE_H
