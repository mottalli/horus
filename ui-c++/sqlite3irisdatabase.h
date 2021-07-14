#ifndef SQLITE3IRISDATABASE_H
#define SQLITE3IRISDATABASE_H

#include <iostream>
#include <QDebug>
#include <QObject>
#include <boost/optional.hpp>

#include "common.h"
#include "sqlite3wrapper.h"

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
	int addUser(string userName, const IrisTemplate& irisTemplate, const SegmentationResult& segmentationResult, const Image& image, horus::VideoProcessor::CaptureBurst captureBurst=horus::VideoProcessor::CaptureBurst());
	int addImage(int userId, const Image& image, const SegmentationResult& segmentationResult, optional<IrisTemplate> averageTemplate = optional<IrisTemplate>(), horus::VideoProcessor::CaptureBurst captureBurst=horus::VideoProcessor::CaptureBurst());

private:
	string dbPath;
	mutable SQlite3Database db;
};

#endif // SQLITE3IRISDATABASE_H
