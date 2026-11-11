/* libsqlite3-sys's loadable wrapper drops variadic arguments. */
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT3

int vec_lance_constraint_support(sqlite3 *db) {
  return sqlite3_vtab_config(db, SQLITE_VTAB_CONSTRAINT_SUPPORT, 1);
}
