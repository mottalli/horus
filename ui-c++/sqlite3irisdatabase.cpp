#include <sys/stat.h>
#include "sqlite3irisdatabase.h"
#include "processingthread.h"

extern ProcessingThread PROCESSING_THREAD;

SQLite3IrisDatabase::SQLite3IrisDatabase(const string& dbPath) :
	dbPath(dbPath), db(NULL)
{
	string sql;
	string dbFile = dbPath + "/base.db";
	sqlite3_stmt* rows;

	VERIFY_SQL( sqlite3_open(dbFile.c_str(), &this->db), "Could not open database file " + dbFile );

	qDebug() << "Cargando base de datos...";

	sql = "SELECT id_usuario, iris_template FROM usuarios NATURAL JOIN base_iris WHERE imagen_primaria=1";
	VERIFY_SQL( sqlite3_prepare(this->db, sql.c_str(), -1, &rows, NULL) );
	while (sqlite3_step(rows) == SQLITE_ROW) {
		int idUsuario = sqlite3_column_int(rows, 0);
		string serializedTemplate = (const char*)sqlite3_column_text(rows, 1);

		if (serializedTemplate.length() == 0) {
			throw runtime_error("Se detectó una imagen no codificada");
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

	string sql = "SELECT nombre,segmentacion,iris_template FROM usuarios NATURAL JOIN base_iris WHERE id_usuario=? AND imagen_primaria=1";
	VERIFY_SQL( sqlite3_prepare(this->db, sql.c_str(), -1, &stmt, NULL) );
	VERIFY_SQL( sqlite3_bind_int(stmt, 1, userId) );

	if (sqlite3_step(stmt) == SQLITE_ROW) {
		// Match
		string serializedSegmentation = (const char*)sqlite3_column_text(stmt, 1);
		string serializedTemplate = (const char*)sqlite3_column_text(stmt, 2);
		string imagePath = this->getImagePathForUser(userId);

		res.userId = userId;
		res.userName = (const char*)sqlite3_column_text(stmt, 0);
		res.segmentation = Serializer::unserializeSegmentationResult(serializedSegmentation);
		res.irisTemplate = Serializer::unserializeIrisTemplate(serializedTemplate);
		res.image = imread(imagePath, 1);
	}

	return res;
}

void SQLite3IrisDatabase::addUser(string userName, const IrisTemplate& irisTemplate, const SegmentationResult& segmentationResult, const Image& image)
{
	if (userName.empty()) {
		throw runtime_error("El nombre no puede estar vacío");
	}

	// Me fijo si hay otro con el mismo nombre
	sqlite3_stmt* stmt;
	string sql = "SELECT id_usuario FROM usuarios WHERE LOWER(nombre) = LOWER(?)";
	VERIFY_SQL( sqlite3_prepare(this->db, sql.c_str(), -1, &stmt, NULL) );
	VERIFY_SQL( sqlite3_bind_text(stmt, 1, userName.c_str(), -1, SQLITE_TRANSIENT) );
	bool userExists = (sqlite3_step(stmt) == SQLITE_ROW);
	if (userExists) {
		throw runtime_error("Ya existe un usuario en la base de datos con ese nombre");
	}

	// Inserto
	sql = "INSERT INTO usuarios(nombre) VALUES(?)";
	VERIFY_SQL( sqlite3_prepare(this->db, sql.c_str(), -1, &stmt, NULL) );
	VERIFY_SQL( sqlite3_bind_text(stmt, 1, userName.c_str(), -1, SQLITE_TRANSIENT) );
	if (sqlite3_step(stmt) != SQLITE_DONE) {
		throw runtime_error(string("No se pudo insertar el registro en la base [") + sqlite3_errmsg(this->db) + string("]"));
	}

	// Obtengo el ID insertado
	sqlite3_int64 userId = sqlite3_last_insert_rowid(this->db);

	sqlite3_finalize(stmt);

	// Guardo la imagen
	this->addImage(userId, image, segmentationResult, irisTemplate);

	this->addTemplate(userId, irisTemplate);
}

void SQLite3IrisDatabase::addImage(int userId, const Image& image, const SegmentationResult& segmentationResult, optional<IrisTemplate> averageTemplate)
{
	string filename, sql;
	sqlite3_stmt* stmt;

	for (int i = 1; ; i++) {
		filename = (boost::format("%i_%i.jpg") % userId % i).str();
		string fullFilename = (boost::format("%s/%s") % this->dbPath % filename).str();			// /path/to/db/<id>_<i>.jpg
		if (!filesystem::is_regular_file(fullFilename)) {			// Encontré un nombre disponible para el archivo
			imwrite(fullFilename, image);
			break;
		}
	}

	IrisTemplate imageTemplate = ::PROCESSING_THREAD.videoProcessor.irisEncoder.generateTemplate(image, segmentationResult);

	// Chequeo si es la primer imagen
	sql = "SELECT COUNT(*) FROM base_iris WHERE id_usuario=? AND imagen_primaria=?";
	VERIFY_SQL( sqlite3_prepare(this->db, sql.c_str(), -1, &stmt, NULL) );
	VERIFY_SQL( sqlite3_bind_int(stmt,  1, userId) );
	VERIFY_SQL( sqlite3_bind_int(stmt,  2, 1) );
	if (sqlite3_step(stmt) != SQLITE_ROW) {
		throw runtime_error(string("Error en consulta [") + sqlite3_errmsg(this->db) + "]");
	}
	int imagenPrimaria = (sqlite3_column_int(stmt, 0) == 0) ? 1 : 0;
	sqlite3_finalize(stmt);

	string serializedImageTemplate = Serializer::serializeIrisTemplate(imageTemplate);
	string serializedAverageTemplate = ( (averageTemplate) ? Serializer::serializeIrisTemplate(*averageTemplate) : "");
	string serializedSegmentationResult = Serializer::serializeSegmentationResult(segmentationResult);

	sql = "INSERT INTO base_iris(id_usuario,imagen_primaria,imagen,segmentacion,segmentacion_correcta,iris_template,average_template) \
						VALUES (?,?,?,?,?,?,?)";
	VERIFY_SQL( sqlite3_prepare(this->db, sql.c_str(), -1, &stmt, NULL) );
	VERIFY_SQL( sqlite3_bind_int(stmt,  1, userId) );
	VERIFY_SQL( sqlite3_bind_int(stmt,  2, imagenPrimaria) );
	VERIFY_SQL( sqlite3_bind_text(stmt, 3, filename.c_str(), -1, SQLITE_TRANSIENT) );
	VERIFY_SQL( sqlite3_bind_text(stmt, 4, serializedSegmentationResult.c_str(), -1, SQLITE_TRANSIENT) );
	VERIFY_SQL( sqlite3_bind_int(stmt,  5, 1) );
	VERIFY_SQL( sqlite3_bind_text(stmt, 6, serializedImageTemplate.c_str(), -1, SQLITE_TRANSIENT) );
	VERIFY_SQL( sqlite3_bind_text(stmt, 7, serializedAverageTemplate.c_str(), -1, SQLITE_TRANSIENT) );
	if (sqlite3_step(stmt) != SQLITE_DONE) {
		throw runtime_error(string("No se pudo insertar el registro en la base [") + sqlite3_errmsg(this->db) + string("]"));
	}
	sqlite3_finalize(stmt);
}
