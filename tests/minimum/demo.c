#include "sqlite3.h"
#include "sqlite-vec.h"
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>

int main(int argc, char *argv[]) {
  int rc = SQLITE_OK;
  sqlite3 *db;
  sqlite3_stmt *stmt;

  rc = sqlite3_open(":memory:", &db);
  assert(rc == SQLITE_OK);

  rc = sqlite3_vec_init(db, NULL, NULL);
  assert(rc == SQLITE_OK);


  rc = sqlite3_prepare_v2(db, "SELECT sqlite_version(), vec_version(), (select json_group_array(compile_options) from pragma_compile_options)", -1, &stmt, NULL);
  assert(rc == SQLITE_OK);

  rc = sqlite3_step(stmt);
  printf("sqlite_version=%s, vec_version=%s %s\n", sqlite3_column_text(stmt, 0), sqlite3_column_text(stmt, 1), sqlite3_column_text(stmt, 2));
  sqlite3_finalize(stmt);

  rc = sqlite3_exec(db,
    "CREATE VIRTUAL TABLE v USING vec0(e float[2]);"
    "INSERT INTO v(rowid, e) VALUES (1, '[0,1]');"
    "CREATE TABLE edges(src INTEGER, dst INTEGER);"
    "INSERT INTO edges VALUES (1, 2);", NULL, NULL, NULL);
  assert(rc == SQLITE_OK);
  rc = sqlite3_prepare_v2(db, "SELECT rowid FROM v WHERE e MATCH '[0,1]' AND k = 1", -1, &stmt, NULL);
  assert(rc == SQLITE_OK);
  assert(sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_int64(stmt, 0) == 1);
  sqlite3_finalize(stmt);

  // graph_bfs needs SQLite 3.37+; older SQLite reports that instead of crashing.
  rc = sqlite3_prepare_v2(db, "SELECT node FROM graph_bfs('edges', 'src', 'dst', 1, 1)", -1, &stmt, NULL);
  if (rc == SQLITE_OK) rc = sqlite3_step(stmt);
  printf("graph_bfs: rc=%d %s\n", rc, sqlite3_errmsg(db));
  if (sqlite3_libversion_number() < 3037000) {
    assert(rc == SQLITE_ERROR && strstr(sqlite3_errmsg(db), "3.37.0") != NULL);
  } else {
    assert(rc == SQLITE_ROW);
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return 0;
}
