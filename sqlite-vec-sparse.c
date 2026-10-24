/**
 * sqlite-vec-sparse.c — sparse vectors (SPLADE, BM25-style term weights).
 *
 * #include'd into sqlite-vec.c after vec_each and before sqlite3_vec_init().
 *
 * Format: a BLOB of (u32 index, f32 value) pairs, sorted by index, unique,
 * zeros dropped. Little-endian, like every other sqlite-vec vector.
 *
 * Indexing is left to SQLite itself: vec_sparse_each() turns a sparse vector
 * into (idx, value) rows, so an inverted index is a plain WITHOUT ROWID table
 * keyed on (term, doc) and a query is a join + GROUP BY. See reference.yaml.
 */

#ifndef SQLITE_VEC_SPARSE_C
#define SQLITE_VEC_SPARSE_C

#ifndef SQLITE_VEC_H
#include "sqlite-vec.c" // IWYU pragma: keep
#endif

struct SparseEntry {
  u32 idx;
  f32 value;
};

#define SPARSE_ENTRY_SIZE 8

static int sparse_entry_cmp(const void *a, const void *b) {
  u32 x = ((const struct SparseEntry *)a)->idx;
  u32 y = ((const struct SparseEntry *)b)->idx;
  return (x > y) - (x < y);
}

static void sparse_read(const u8 *blob, int i, u32 *idx, f32 *value) {
  memcpy(idx, blob + i * SPARSE_ENTRY_SIZE, sizeof(u32));
  memcpy(value, blob + i * SPARSE_ENTRY_SIZE + sizeof(u32), sizeof(f32));
}

/**
 * Validate a sparse BLOB: whole entries, strictly increasing indexes.
 * Returns the entry count, or -1 when malformed.
 */
static int sparse_validate(const u8 *blob, int nBytes) {
  if (nBytes % SPARSE_ENTRY_SIZE != 0) return -1;
  int n = nBytes / SPARSE_ENTRY_SIZE;
  u32 prev = 0;
  for (int i = 0; i < n; i++) {
    u32 idx;
    f32 value;
    sparse_read(blob, i, &idx, &value);
    if (i > 0 && idx <= prev) return -1;
    prev = idx;
  }
  return n;
}

static int sparse_from_value(sqlite3_context *context, sqlite3_value *value,
                             const u8 **blob, int *n) {
  if (sqlite3_value_type(value) != SQLITE_BLOB &&
      sqlite3_value_type(value) != SQLITE_NULL) {
    sqlite3_result_error(context, "sparse vector must be a BLOB from vec_sparse()", -1);
    return SQLITE_ERROR;
  }
  *blob = sqlite3_value_blob(value);
  *n = sparse_validate(*blob, sqlite3_value_bytes(value));
  if (*n < 0) {
    sqlite3_result_error(context, "malformed sparse vector", -1);
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

static void sparse_result(sqlite3_context *context, struct SparseEntry *e,
                          int n) {
  qsort(e, n, sizeof(*e), sparse_entry_cmp);
  int out = 0;
  for (int i = 0; i < n; i++) {
    if (e[i].value == 0.0f) continue;
    if (out > 0 && e[out - 1].idx == e[i].idx) {
      sqlite3_free(e);
      sqlite3_result_error(context, "duplicate index in sparse vector", -1);
      return;
    }
    e[out++] = e[i];
  }
  u8 *blob = sqlite3_malloc(out * SPARSE_ENTRY_SIZE + 1);
  if (!blob) {
    sqlite3_free(e);
    sqlite3_result_error_nomem(context);
    return;
  }
  for (int i = 0; i < out; i++) {
    memcpy(blob + i * SPARSE_ENTRY_SIZE, &e[i].idx, sizeof(u32));
    memcpy(blob + i * SPARSE_ENTRY_SIZE + sizeof(u32), &e[i].value, sizeof(f32));
  }
  sqlite3_free(e);
  sqlite3_result_blob(context, blob, out * SPARSE_ENTRY_SIZE, sqlite3_free);
}

/**
 * Parse a JSON object of {"index": weight, ...}.
 */
static int sparse_parse_json_object(const char *s, struct SparseEntry **out,
                                    int *nOut, const char **err) {
  int cap = 16, n = 0;
  struct SparseEntry *e = sqlite3_malloc(cap * sizeof(*e));
  if (!e) { *err = "out of memory"; return SQLITE_NOMEM; }
  const char *p = s;
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
  if (*p++ != '{') goto malformed;
  for (;;) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
    if (*p == '}') break;
    if (*p++ != '"') goto malformed;
    char *end;
    errno = 0;
    unsigned long long idx = strtoull(p, &end, 10);
    if (end == p || *end != '"' || errno || idx > UINT32_MAX || *p == '-') goto malformed;
    p = end + 1;
    while (*p == ' ') p++;
    if (*p++ != ':') goto malformed;
    while (*p == ' ') p++;
    double v = strtod_c(p, &end);
    if (end == p) goto malformed;
    p = end;
    if (n == cap) {
      cap *= 2;
      struct SparseEntry *tmp = sqlite3_realloc64(e, cap * sizeof(*e));
      if (!tmp) { sqlite3_free(e); *err = "out of memory"; return SQLITE_NOMEM; }
      e = tmp;
    }
    e[n].idx = (u32)idx;
    e[n].value = (f32)v;
    n++;
  }
  *out = e;
  *nOut = n;
  return SQLITE_OK;
malformed:
  sqlite3_free(e);
  *err = "sparse vector JSON must be an object like {\"12\": 0.5, \"907\": 1.25}";
  return SQLITE_ERROR;
}

/**
 * vec_sparse(x): x is a JSON object {"idx": weight} or any dense float32
 * vector input (JSON array or BLOB), whose nonzero elements are kept.
 */
static void vec_sparse(sqlite3_context *context, int argc, sqlite3_value **argv) {
  assert(argc == 1);
  struct SparseEntry *e = NULL;
  int n = 0;
  if (sqlite3_value_type(argv[0]) == SQLITE_TEXT) {
    const char *s = (const char *)sqlite3_value_text(argv[0]);
    const char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p == '{') {
      const char *err;
      int rc = sparse_parse_json_object(s, &e, &n, &err);
      if (rc == SQLITE_NOMEM) { sqlite3_result_error_nomem(context); return; }
      if (rc != SQLITE_OK) { sqlite3_result_error(context, err, -1); return; }
      sparse_result(context, e, n);
      return;
    }
  }
  void *vector;
  size_t dims;
  enum VectorElementType type;
  vector_cleanup cleanup;
  char *err;
  if (vector_from_value(argv[0], &vector, &dims, &type, &cleanup, &err) != SQLITE_OK) {
    sqlite3_result_error(context, err, -1);
    sqlite3_free(err);
    return;
  }
  if (type != SQLITE_VEC_ELEMENT_TYPE_FLOAT32) {
    cleanup(vector);
    sqlite3_result_error(context, "vec_sparse() only converts float32 vectors", -1);
    return;
  }
  e = sqlite3_malloc64((dims ? dims : 1) * sizeof(*e));
  if (!e) { cleanup(vector); sqlite3_result_error_nomem(context); return; }
  for (size_t i = 0; i < dims; i++) {
    f32 v = ((f32 *)vector)[i];
    if (v != 0.0f) { e[n].idx = (u32)i; e[n].value = v; n++; }
  }
  cleanup(vector);
  sparse_result(context, e, n);
}

/**
 * Merge-join over two sorted sparse vectors: dot, and both squared norms.
 */
static void sparse_merge(const u8 *a, int na, const u8 *b, int nb, double *dot,
                         double *aa, double *bb) {
  int i = 0, j = 0;
  *dot = *aa = *bb = 0;
  u32 ia, ib;
  f32 va, vb;
  for (int k = 0; k < na; k++) { sparse_read(a, k, &ia, &va); *aa += (double)va * va; }
  for (int k = 0; k < nb; k++) { sparse_read(b, k, &ib, &vb); *bb += (double)vb * vb; }
  while (i < na && j < nb) {
    sparse_read(a, i, &ia, &va);
    sparse_read(b, j, &ib, &vb);
    if (ia == ib) { *dot += (double)va * vb; i++; j++; }
    else if (ia < ib) i++;
    else j++;
  }
}

static void vec_sparse_distance(sqlite3_context *context, int argc,
                                sqlite3_value **argv, int cosine) {
  assert(argc == 2);
  const u8 *a, *b;
  int na, nb;
  if (sparse_from_value(context, argv[0], &a, &na) != SQLITE_OK) return;
  if (sparse_from_value(context, argv[1], &b, &nb) != SQLITE_OK) return;
  double dot, aa, bb;
  sparse_merge(a, na, b, nb, &dot, &aa, &bb);
  if (!cosine) {
    sqlite3_result_double(context, -dot);
  } else if (aa == 0 || bb == 0) {
    sqlite3_result_null(context);
  } else {
    sqlite3_result_double(context, 1.0 - dot / (sqrt(aa) * sqrt(bb)));
  }
}

static void vec_sparse_distance_dot(sqlite3_context *context, int argc,
                                    sqlite3_value **argv) {
  vec_sparse_distance(context, argc, argv, 0);
}

static void vec_sparse_distance_cosine(sqlite3_context *context, int argc,
                                       sqlite3_value **argv) {
  vec_sparse_distance(context, argc, argv, 1);
}

static void vec_sparse_to_json(sqlite3_context *context, int argc,
                               sqlite3_value **argv) {
  assert(argc == 1);
  const u8 *a;
  int n;
  if (sparse_from_value(context, argv[0], &a, &n) != SQLITE_OK) return;
  sqlite3_str *s = sqlite3_str_new(NULL);
  sqlite3_str_appendchar(s, 1, '{');
  for (int i = 0; i < n; i++) {
    u32 idx;
    f32 v;
    sparse_read(a, i, &idx, &v);
    sqlite3_str_appendf(s, "%s\"%u\":%!.9g", i ? "," : "", idx, (double)v);
  }
  sqlite3_str_appendchar(s, 1, '}');
  int len = sqlite3_str_length(s);
  char *z = sqlite3_str_finish(s);
  if (!z) { sqlite3_result_error_nomem(context); return; }
  sqlite3_result_text(context, z, len, sqlite3_free);
}

#pragma region vec_sparse_each table function

typedef struct vec_sparse_each_cursor vec_sparse_each_cursor;
struct vec_sparse_each_cursor {
  sqlite3_vtab_cursor base;
  i64 iRowid;
  u8 *blob;
  int n;
};

static int vec_sparse_eachConnect(sqlite3 *db, void *pAux, int argc,
                                  const char *const *argv,
                                  sqlite3_vtab **ppVtab, char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  int rc = sqlite3_declare_vtab(db, "CREATE TABLE x(idx, value, vector hidden)");
#define VEC_SPARSE_EACH_COLUMN_IDX 0
#define VEC_SPARSE_EACH_COLUMN_VALUE 1
#define VEC_SPARSE_EACH_COLUMN_VECTOR 2
  if (rc != SQLITE_OK) return rc;
  sqlite3_vtab *pNew = sqlite3_malloc(sizeof(*pNew));
  if (!pNew) return SQLITE_NOMEM;
  memset(pNew, 0, sizeof(*pNew));
  *ppVtab = pNew;
  return SQLITE_OK;
}

static int vec_sparse_eachDisconnect(sqlite3_vtab *pVtab) {
  sqlite3_free(pVtab);
  return SQLITE_OK;
}

static int vec_sparse_eachOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor) {
  UNUSED_PARAMETER(p);
  vec_sparse_each_cursor *pCur = sqlite3_malloc(sizeof(*pCur));
  if (!pCur) return SQLITE_NOMEM;
  memset(pCur, 0, sizeof(*pCur));
  *ppCursor = &pCur->base;
  return SQLITE_OK;
}

static int vec_sparse_eachClose(sqlite3_vtab_cursor *cur) {
  vec_sparse_each_cursor *pCur = (vec_sparse_each_cursor *)cur;
  sqlite3_free(pCur->blob);
  sqlite3_free(pCur);
  return SQLITE_OK;
}

static int vec_sparse_eachBestIndex(sqlite3_vtab *pVTab,
                                    sqlite3_index_info *pIdxInfo) {
  UNUSED_PARAMETER(pVTab);
  for (int i = 0; i < pIdxInfo->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &pIdxInfo->aConstraint[i];
    if (c->iColumn == VEC_SPARSE_EACH_COLUMN_VECTOR &&
        c->op == SQLITE_INDEX_CONSTRAINT_EQ && c->usable) {
      pIdxInfo->aConstraintUsage[i].argvIndex = 1;
      pIdxInfo->aConstraintUsage[i].omit = 1;
      pIdxInfo->estimatedCost = 100.0;
      pIdxInfo->estimatedRows = 100;
      return SQLITE_OK;
    }
  }
  return SQLITE_CONSTRAINT;
}

static int vec_sparse_eachFilter(sqlite3_vtab_cursor *pVtabCursor, int idxNum,
                                 const char *idxStr, int argc,
                                 sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  assert(argc == 1);
  vec_sparse_each_cursor *pCur = (vec_sparse_each_cursor *)pVtabCursor;
  sqlite3_free(pCur->blob);
  pCur->blob = NULL;
  pCur->n = 0;
  pCur->iRowid = 0;
  int nBytes = sqlite3_value_bytes(argv[0]);
  const u8 *src = sqlite3_value_blob(argv[0]);
  int n = sparse_validate(src, nBytes);
  if (n < 0 || (sqlite3_value_type(argv[0]) != SQLITE_BLOB &&
                sqlite3_value_type(argv[0]) != SQLITE_NULL)) {
    sqlite3_free(pVtabCursor->pVtab->zErrMsg);
    pVtabCursor->pVtab->zErrMsg = sqlite3_mprintf("malformed sparse vector");
    return SQLITE_ERROR;
  }
  if (n > 0) {
    pCur->blob = sqlite3_malloc(nBytes);
    if (!pCur->blob) return SQLITE_NOMEM;
    memcpy(pCur->blob, src, nBytes);
  }
  pCur->n = n;
  return SQLITE_OK;
}

static int vec_sparse_eachRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  *pRowid = ((vec_sparse_each_cursor *)cur)->iRowid;
  return SQLITE_OK;
}

static int vec_sparse_eachEof(sqlite3_vtab_cursor *cur) {
  vec_sparse_each_cursor *pCur = (vec_sparse_each_cursor *)cur;
  return pCur->iRowid >= pCur->n;
}

static int vec_sparse_eachNext(sqlite3_vtab_cursor *cur) {
  ((vec_sparse_each_cursor *)cur)->iRowid++;
  return SQLITE_OK;
}

static int vec_sparse_eachColumn(sqlite3_vtab_cursor *cur,
                                 sqlite3_context *context, int i) {
  vec_sparse_each_cursor *pCur = (vec_sparse_each_cursor *)cur;
  u32 idx;
  f32 value;
  sparse_read(pCur->blob, (int)pCur->iRowid, &idx, &value);
  if (i == VEC_SPARSE_EACH_COLUMN_IDX) sqlite3_result_int64(context, idx);
  else if (i == VEC_SPARSE_EACH_COLUMN_VALUE) sqlite3_result_double(context, value);
  return SQLITE_OK;
}

static sqlite3_module vec_sparse_eachModule = {
    /* iVersion    */ 0,
    /* xCreate     */ 0,
    /* xConnect    */ vec_sparse_eachConnect,
    /* xBestIndex  */ vec_sparse_eachBestIndex,
    /* xDisconnect */ vec_sparse_eachDisconnect,
    /* xDestroy    */ 0,
    /* xOpen       */ vec_sparse_eachOpen,
    /* xClose      */ vec_sparse_eachClose,
    /* xFilter     */ vec_sparse_eachFilter,
    /* xNext       */ vec_sparse_eachNext,
    /* xEof        */ vec_sparse_eachEof,
    /* xColumn     */ vec_sparse_eachColumn,
    /* xRowid      */ vec_sparse_eachRowid,
    /* xUpdate     */ 0,
    /* xBegin      */ 0,
    /* xSync       */ 0,
    /* xCommit     */ 0,
    /* xRollback   */ 0,
    /* xFindMethod */ 0,
    /* xRename     */ 0,
    /* xSavepoint  */ 0,
    /* xRelease    */ 0,
    /* xRollbackTo */ 0,
    /* xShadowName */ 0,
#if SQLITE_VERSION_NUMBER >= 3044000
    /* xIntegrity  */ 0
#endif
};

#pragma endregion

/**
 * vec_rrf(rank1, rank2, ...): Reciprocal Rank Fusion score, sum of
 * 1 / (60 + rank) over the non-NULL 1-based ranks. Fuses vec0 KNN ranks with
 * FTS5 bm25() ranks (or any other ranked lists) in plain SQL.
 * ponytail: k is fixed at 60 (the paper's value); add a vec_rrf_k() variant
 * if callers need to tune it.
 */
static void vec_rrf(sqlite3_context *context, int argc, sqlite3_value **argv) {
  double score = 0;
  for (int i = 0; i < argc; i++) {
    if (sqlite3_value_type(argv[i]) == SQLITE_NULL) continue;
    double rank = sqlite3_value_double(argv[i]);
    if (rank < 1) {
      sqlite3_result_error(context, "vec_rrf() ranks start at 1", -1);
      return;
    }
    score += 1.0 / (60.0 + rank);
  }
  sqlite3_result_double(context, score);
}

#endif /* SQLITE_VEC_SPARSE_C */
