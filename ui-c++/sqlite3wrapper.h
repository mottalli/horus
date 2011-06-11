#pragma once

#include "common.h"

#include "external/sqlite3/sqlite3.h"

class SQLException : public runtime_error
{
public:
	SQLException(const string& msg) : runtime_error(msg) {}
};

void _VERIFY_SQL(int status, sqlite3* db, string msg="");

class SQlite3Database
{
public:
	SQlite3Database() : db(NULL) {}

	void open(const string& fullPath_)
	{
		this->fullPath = fullPath_;
		_VERIFY_SQL( sqlite3_open(this->fullPath.c_str(), &this->db), this->db, "Could not open database file " + fullPath );
		assert(this->db != NULL);
	}

	/*~SQlite3Database()
	{
		if (this->db) {
			sqlite3_close(this->db);
		}
	}*/

	string lastErrorMsg() { return (const char*)sqlite3_errmsg(this->db); }
	int lastInsertRowid() { return sqlite3_last_insert_rowid(this->db); }

	class Recordset
	{
	public:
		Recordset(sqlite3_stmt* stmt_) : stmt(stmt_), available(false) {}
		bool next()
		{
			int res = sqlite3_step(this->stmt);
			if (res == SQLITE_ROW) {
				this->available = true;
				return true;
			} else if (res == SQLITE_DONE) {
				this->available = false;
				return false;
			} else {
				this->available = false;
				string error = string("Error while fetching row: ") + sqlite3_errmsg(sqlite3_db_handle(this->stmt));
				throw SQLException(error);
			}
		}

		template<class T> T at(int idx) { throw runtime_error("Unsupported column type"); }
		inline bool isAvailable() { return this->available; }
	private:
		sqlite3_stmt* stmt;
		bool available;
	};

	class PreparedStatement
	{
	public:
		PreparedStatement(sqlite3_stmt* stmt_) : stmt(stmt_), bindIdx(1) { }
		//~PreparedStatement() { sqlite3_finalize(this->stmt); }
		Recordset getRecordset() { return Recordset(this->stmt); }
		Recordset getOne() { Recordset rs(this->stmt); rs.next(); return rs; }
		void run() { Recordset rs(this->stmt); rs.next(); }
		PreparedStatement& operator<<(int n) { sqlite3_bind_int(this->stmt, this->bindIdx++, n); return *this; }
		PreparedStatement& operator<<(const string& str) { sqlite3_bind_text(this->stmt, this->bindIdx++, str.c_str(), -1, SQLITE_TRANSIENT); return *this; }
	private:
		sqlite3_stmt* stmt;
		int bindIdx;
	};

	PreparedStatement prepareStatement(const string& sql)
	{
		sqlite3_stmt* pstmt;
		_VERIFY_SQL( sqlite3_prepare(this->db, sql.c_str(), -1, &pstmt, NULL), this->db, "Could not prepare SQL: " + sql );
		return PreparedStatement(pstmt);
	}

private:
	string fullPath;
	sqlite3* db;
};

template<> int SQlite3Database::Recordset::at(int idx);
template<> string SQlite3Database::Recordset::at(int idx);
