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
		string serializedTemplate = string( (const char*)sqlite3_column_text(rows, 1) );

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
		cout << status << endl;
		throw runtime_error(msgError + " [" + sqlite3_errmsg(this->db) + "]");
	}
}
