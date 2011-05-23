#include "sqlite3wrapper.h"

void _VERIFY_SQL(int status, sqlite3* db, std::string msg)
{
	if (status != SQLITE_OK) {
		if (!msg.empty()) {
			msg = msg + "[" + sqlite3_errmsg(db) + "]";
		} else {
			msg = sqlite3_errmsg(db);
		}
		throw SQLException(msg);
	}
}

template<> int SQlite3Database::Recordset::at(int idx)
{
	return sqlite3_column_int(this->stmt, idx);
}

template<> std::string SQlite3Database::Recordset::at(int idx)
{
	return (const char*)sqlite3_column_text(this->stmt, idx);
}
