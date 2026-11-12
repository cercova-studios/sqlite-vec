#ifndef SQLITE_VEC_H
#define SQLITE_VEC_H

#ifndef SQLITE_CORE
#include "sqlite3ext.h"
#else
#include "sqlite3.h"
#endif

#ifdef SQLITE_VEC_STATIC
  #define SQLITE_VEC_API
#else
  #ifdef _WIN32
    #define SQLITE_VEC_API __declspec(dllexport)
  #else
    #define SQLITE_VEC_API
  #endif
#endif

#define SQLITE_VEC_VERSION "v0.1.10-alpha.4"
// TODO rm
#define SQLITE_VEC_DATE "2026-10-24T13:40:37Z-0400"
#define SQLITE_VEC_SOURCE "04d28bd21773981e2d266bbf6aa4efbd011eb4f6"


#define SQLITE_VEC_VERSION_MAJOR 0
#define SQLITE_VEC_VERSION_MINOR 1
#define SQLITE_VEC_VERSION_PATCH 10

#ifdef __cplusplus
extern "C" {
#endif

SQLITE_VEC_API int sqlite3_vec_init(sqlite3 *db, char **pzErrMsg,
                  const sqlite3_api_routines *pApi);

/*
** Storage providers: where a vec0 table keeps its vectors and auxiliary
** (+name type) columns. Without one, vectors live in vec0's shadow tables and
** auxiliary columns in <table>_auxiliary. Indexes (an HNSW graph, for
** example) stay in SQLite either way.
**
**   create virtual table t using vec0(
**     v float[768] indexed by hnsw(), +payload blob,
**     storage=lance, storage_uri='/data/t.lance');
**
** A provider registers itself on a connection, before tables that use it are
** opened, with
**
**   sqlite3_set_clientdata(db, SQLITE_VEC_STORAGE_KEY_PREFIX "lance", pMethods, 0);
**
** pMethods must outlive the connection. Every method returns an SQLite result
** code and, on failure, may leave a message for xErrmsg.
**
** Transactions: vec0 forwards its xSync, xCommit, xRollback, xSavepoint,
** xRelease and xRollbackTo to the provider. A provider must make writes
** durable no earlier than xSync, must undo writes made after savepoint N on
** xRollbackTo(N) (SQLite uses this when one statement fails inside a
** transaction), and must let reads see the transaction's own writes.
*/
#define SQLITE_VEC_STORAGE_KEY_PREFIX "sqlite-vec-storage:"
#define SQLITE_VEC_STORAGE_VERSION 2

typedef struct sqlite3_vec_storage sqlite3_vec_storage; /* one per vec0 table */

typedef struct sqlite3_vec_storage_methods {
  int iVersion; /* SQLITE_VEC_STORAGE_VERSION */

  /* Open the store of table zSchema.zTable. bCreate is 1 for CREATE VIRTUAL
  ** TABLE, 0 when an existing table is opened. zUri is storage_uri, or NULL.
  **
  ** Columns are numbered auxiliary first, then vectors. Auxiliary column i
  ** is named azAux[i] with SQLite type aAuxType[i] (SQLITE_INTEGER,
  ** SQLITE_FLOAT, SQLITE_TEXT or SQLITE_BLOB), and may be NULL. Vector
  ** column j (column nAux + j) is named azVec[j] and holds aVecBytes[j]
  ** bytes; aVecDims[j] is its dimension count for a float32 vector, which a
  ** provider may store in a searchable form, and 0 for other element types. */
  int (*xOpen)(sqlite3 *db, const char *zSchema, const char *zTable,
               const char *zUri, int bCreate,
               int nAux, const char *const *azAux, const int *aAuxType,
               int nVec, const char *const *azVec, const int *aVecBytes,
               const int *aVecDims,
               sqlite3_vec_storage **ppStore, char **pzErr);
  void (*xClose)(sqlite3_vec_storage *);
  /* Transactional DROP, before vec0 removes its shadow tables. May reject
  ** without discarding pending state. Retain external data needed by rollback
  ** or readers; xClose releases the handle. */
  int (*xDestroy)(sqlite3_vec_storage *);
  int (*xRename)(sqlite3_vec_storage *, const char *zNewTable);

  /* Writes. xPut stores a new row: nAux auxiliary values and nVec vectors
  ** (raw bytes, aVecBytes[j] each). xSet replaces auxiliary column iCol of an
  ** existing row; xSetVector replaces vector j. */
  int (*xPut)(sqlite3_vec_storage *, sqlite3_int64 rowid, sqlite3_value **apAux,
              const void *const *apVec);
  int (*xSet)(sqlite3_vec_storage *, sqlite3_int64 rowid, int iCol, sqlite3_value *);
  int (*xSetVector)(sqlite3_vec_storage *, sqlite3_int64 rowid, int iVec, const void *pVec);
  int (*xDelete)(sqlite3_vec_storage *, sqlite3_int64 rowid);

  /* Reads, as the current transaction sees the data.
  **
  ** xColumn sets column iCol of rowid as the result of ctx (NULL when the row
  ** has no value). xRead sets *ppData and *pnData to the bytes of column iCol,
  ** valid until the next call on the store, or *ppData to NULL when there is
  ** no such row. xScan calls xCallback(pArg, rowid, data, n) for column iCol
  ** of every row, stopping with the callback's result when it is not
  ** SQLITE_OK. xPrefetch hints that these rowids are about to be read, so a
  ** provider can fetch them in one batch. */
  int (*xColumn)(sqlite3_vec_storage *, sqlite3_int64 rowid, int iCol,
                 sqlite3_context *ctx);
  int (*xRead)(sqlite3_vec_storage *, sqlite3_int64 rowid, int iCol,
               const void **ppData, int *pnData);
  int (*xScan)(sqlite3_vec_storage *, int iCol,
               int (*xCallback)(void *pArg, sqlite3_int64 rowid, const void *pData, int nData),
               void *pArg);
  int (*xPrefetch)(sqlite3_vec_storage *, const sqlite3_int64 *aRowid, int n);

  int (*xSync)(sqlite3_vec_storage *);
  int (*xCommit)(sqlite3_vec_storage *);
  int (*xRollback)(sqlite3_vec_storage *);
  int (*xSavepoint)(sqlite3_vec_storage *, int iSavepoint);
  int (*xRelease)(sqlite3_vec_storage *, int iSavepoint);
  int (*xRollbackTo)(sqlite3_vec_storage *, int iSavepoint);

  const char *(*xErrmsg)(sqlite3_vec_storage *);

  /* A table command, from insert into t(t) values ('<command>'): vec0
  ** forwards the commands it does not handle itself. Returns SQLITE_EMPTY
  ** for commands the provider does not know. */
  int (*xCommand)(sqlite3_vec_storage *, const char *zCommand);
} sqlite3_vec_storage_methods;

#ifdef __cplusplus
}  /* end of the 'extern "C"' block */
#endif

#endif /* ifndef SQLITE_VEC_H */
