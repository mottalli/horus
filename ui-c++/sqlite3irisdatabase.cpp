#include "sqlite3irisdatabase.h"
#include <sys/stat.h>

#include "external/boost/lexical_cast.hpp"
#include "external/boost/system/error_code.hpp"
#include "external/boost/filesystem.hpp"

SQLite3IrisDatabase::SQLite3IrisDatabase(const string& dbPath) :
	dbPath(dbPath), db(NULL)
{
	string sql;
	string dbFile = dbPath + "/base.db";
	sqlite3_stmt* rows;

	VERIFY_SQL( sqlite3_open(dbFile.c_str(), &this->db), "Could not open database file " + dbFile );

	qDebug() << "Cargando base de datos...";

	sql = "SELECT id_usuario, codigo_gabor FROM usuarios";
	VERIFY_SQL( sqlite3_prepare(this->db, sql.c_str(), -1, &rows, NULL) );
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

void SQLite3IrisDatabase::VERIFY_SQL(int status, const string msgError) const
{
	if (status != SQLITE_OK) {
		throw runtime_error(msgError + " [" + sqlite3_errmsg(this->db) + "]");
	}
}

SQLite3IrisDatabase::IrisData SQLite3IrisDatabase::getIrisData(int userId) const
{
	IrisData res;
	sqlite3_stmt* stmt;

	res.userId = -1;

	string sql = "SELECT nombre,segmentacion,codigo_gabor,imagen FROM usuarios WHERE id_usuario=?";
	VERIFY_SQL( sqlite3_prepare(this->db, sql.c_str(), -1, &stmt, NULL) );
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

void SQLite3IrisDatabase::addUser(string userName, const IrisTemplate& irisTemplate, const SegmentationResult& segmentationResult, const Mat image)
{
	if (userName.empty()) {
		throw std::runtime_error("El nombre no puede estar vacío");
	}

	// Me fijo si hay otro con el mismo nombre
	sqlite3_stmt* stmt;
	string sql = "SELECT id_usuario FROM usuarios WHERE LOWER(nombre) = LOWER(?)";
	VERIFY_SQL( sqlite3_prepare(this->db, sql.c_str(), -1, &stmt, NULL) );
	VERIFY_SQL( sqlite3_bind_text(stmt, 1, userName.c_str(), -1, SQLITE_TRANSIENT) );
	bool userExists = (sqlite3_step(stmt) == SQLITE_ROW);
	if (userExists) {
		throw std::runtime_error("Ya existe un usuario en la base de datos con ese nombre");
	}

	string serializedTemplate = Serializer::serializeIrisTemplate(irisTemplate);
	string serializedSegmentation = Serializer::serializeSegmentationResult(segmentationResult);
	string fullImagePath = "-x-";		// Valor temporario

	// Inserto
	sql = "INSERT INTO usuarios(nombre,imagen,segmentacion,codigo_gabor VALUES(?,?,?,?)";
	VERIFY_SQL( sqlite3_prepare(this->db, sql.c_str(), -1, &stmt, NULL) );
	VERIFY_SQL( sqlite3_bind_text(stmt, 1, userName.c_str(), -1, SQLITE_TRANSIENT) );
	VERIFY_SQL( sqlite3_bind_text(stmt, 2, fullImagePath.c_str(), -1, SQLITE_TRANSIENT) );
	VERIFY_SQL( sqlite3_bind_text(stmt, 3, serializedSegmentation.c_str(), -1, SQLITE_TRANSIENT) );
	VERIFY_SQL( sqlite3_bind_text(stmt, 4, serializedTemplate.c_str(), -1, SQLITE_TRANSIENT) );
	if (sqlite3_step(stmt) != SQLITE_DONE) {
		throw std::runtime_error(string("No se pudo insertar el registro en la base [") + boost::lexical_cast<string>(sqlite3_errmsg(this->db)) + string("]"));
	}

	// Obtengo el ID insertado
	sqlite3_int64 userId = sqlite3_last_insert_rowid(this->db);

	// Guardo la imagen
	/*if (!image.empty()) {
		ostringstream tmpstream;
		string filename = boost::lexical_cast<string>(userId) + ".jpg";
		string relativePath = boost::lexical_cast<string>(userId) + "/";
		string relativeFilename = relativePath + filename;
		string fullFilename = this->dbPath + "/" + relativeFilename;

		if (mkdir(relativePath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
			qDebug() << "No se pudo crear el directorio " << relativePath.c_str() << "!";
		}

		imwrite(fullFilename, image);

		sql = "UPDATE usuarios SET imagen=? WHERE id_usuario=?";
		VERIFY_SQL( sqlite3_prepare(this->db, sql.c_str(), -1, &stmt, NULL) );
		VERIFY_SQL( sqlite3_bind_text(stmt, 1, relativeFilename.c_str(), -1, SQLITE_TRANSIENT) );
		VERIFY_SQL( sqlite3_bind_int(stmt, 2, userId) );
		sqlite3_step(stmt);
	}*/

	this->addTemplate(userId, irisTemplate);
}
