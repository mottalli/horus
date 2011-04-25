#include "sqlite3irisdatabase.h"

SQLite3IrisDatabase::SQLite3IrisDatabase(const string& dbPath) :
	dbPath(dbPath), db(NULL)
{
	string sql;
	string dbFile = dbPath + "/base.db";
	sqlite3_stmt* rows;

	VERIFY_SQL( sqlite3_open(dbFile.c_str(), &this->db), "Could not open database file " + dbFile );

	qDebug() << "Cargando base de datos...";

	sql = "SELECT id_usuario, codigo_gabor FROM usuarios";
	VERIFY_SQL( sqlite3_prepare(this->db, sql.c_str(), sql.length(), &rows, NULL) );
	while (sqlite3_step(rows) == SQLITE_ROW) {
		int idUsuario = sqlite3_column_int(rows, 0);
		string serializedTemplate = (const char*)sqlite3_column_text(rows, 1);

		if (serializedTemplate.length() == 0) {
			throw std::runtime_error("Se detectó una imagen no codificada");
		}

		this->addTemplate(idUsuario, Serializer::unserializeIrisTemplate(serializedTemplate));
	}

	qDebug() << "Fin carga";

}

SQLite3IrisDatabase::~SQLite3IrisDatabase()
{
	if (this->db) {
		sqlite3_close(this->db);
		this->db = NULL;
	}
}

void SQLite3IrisDatabase::VERIFY_SQL(int status, const string msgError)
{
	if (status != SQLITE_OK) {
		throw runtime_error(msgError + " [" + sqlite3_errmsg(this->db) + "]");
	}
}

const SQLite3IrisDatabase::IrisData SQLite3IrisDatabase::getIrisData(int userId)
{
	IrisData res;
	sqlite3_stmt* stmt;

	res.userId = -1;

	string sql = "SELECT nombre,segmentacion,codigo_gabor,imagen FROM usuarios WHERE id_usuario=?";
	VERIFY_SQL( sqlite3_prepare(this->db, sql.c_str(), sql.size(), &stmt, NULL) );
	VERIFY_SQL( sqlite3_bind_int(stmt, 1, userId) );

	if (sqlite3_step(stmt) == SQLITE_ROW) {
		// Match
		res.userName = (const char*)sqlite3_column_text(stmt, 0);
		string serializedSegmentation = (const char*)sqlite3_column_text(stmt, 1);
		res.segmentation = Serializer::unserializeSegmentationResult(serializedSegmentation);
		string serializedTemplate = (const char*)sqlite3_column_text(stmt, 2);
		res.irisTemplate = Serializer::unserializeIrisTemplate(serializedTemplate);

		string imagePath = (const char*)sqlite3_column_text(stmt, 3);
		imagePath = this->dbPath + "/" + imagePath;
		res.image = imread(imagePath, 1);
	}

	return res;
}
