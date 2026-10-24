/**
 * sqlite-vec-hnsw.c — HNSW (Malkov & Yashunin, 2016) graph index for vec0.
 *
 * #include'd into sqlite-vec.c after struct definitions and before vec0_init().
 *
 *   v float[768] indexed by hnsw(m=16, ef_construction=200, ef_search=64, cache_mb=256)
 *
 * Unlike hnswlib-based extensions, the graph lives in SQLite shadow tables, so
 * it is transactional and crash-safe with no save/load step:
 *
 *   _hnsw_nodesNN (rowid PK, level, vector, UNIQUE(level, rowid))
 *   _hnsw_edgesNN (rowid, level, neighbors BLOB of i64) WITHOUT ROWID
 *
 * The entry point is not stored: it is the node with the highest level, found
 * through the UNIQUE(level, rowid) index, so deleting it needs no fix-up.
 *
 * Speed comes from a per-connection cache laid out like hnswlib's memory
 * index: nodes get dense 32-bit indexes, vectors and level-0 neighbor lists
 * sit in fixed-size chunks (pointers stay valid as the cache grows), and the
 * visited set is a tag array. The cache never outlives the state it mirrors:
 * it is cleared on rollback, on statement rollback (xRollbackTo), and when
 * another connection commits (the pager's data version moved).
 *
 * Deletes repair the graph: every out-neighbor of the deleted node re-selects
 * its neighbors from its remaining list plus the deleted node's list. Edges
 * that pointed at the deleted node from elsewhere are skipped at search time.
 */

#ifndef SQLITE_VEC_HNSW_C
#define SQLITE_VEC_HNSW_C

#ifndef SQLITE_VEC_H
#include "sqlite-vec.c" // IWYU pragma: keep
#endif

#define VEC0_HNSW_DEFAULT_M 16
#define VEC0_HNSW_DEFAULT_EF_CONSTRUCTION 200
#define VEC0_HNSW_DEFAULT_EF_SEARCH 64
#define VEC0_HNSW_DEFAULT_CACHE_MB 256
#define VEC0_HNSW_MAX_M 256
#define VEC0_HNSW_MAX_LEVEL 16
// `rowid in (...)` lists up to this size are ranked exactly instead of
// searched: a selective filter starves a graph search of matches.
#define VEC0_HNSW_EXACT_FILTER_MAX 4096

enum { VEC0_HNSW_QUANTIZER_NONE = 0, VEC0_HNSW_QUANTIZER_INT8 = 1 };

#define VEC0_SHADOW_HNSW_NODES_NAME "\"%w\".\"%w_hnsw_nodes%02d\""
#define VEC0_SHADOW_HNSW_EDGES_NAME "\"%w\".\"%w_hnsw_edges%02d\""

// ============================================================================
// Parser
// ============================================================================

static int vec0_parse_hnsw_options(struct Vec0Scanner *scanner,
                                   struct Vec0HnswConfig *config) {
  struct Vec0Token token;
  config->m = VEC0_HNSW_DEFAULT_M;
  config->ef_construction = VEC0_HNSW_DEFAULT_EF_CONSTRUCTION;
  config->ef_search = VEC0_HNSW_DEFAULT_EF_SEARCH;
  config->cache_mb = VEC0_HNSW_DEFAULT_CACHE_MB;
  config->quantizer = VEC0_HNSW_QUANTIZER_NONE;

  int rc = vec0_scanner_next(scanner, &token);
  if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_LPAREN)
    return SQLITE_ERROR;
  rc = vec0_scanner_next(scanner, &token);
  if (rc == VEC0_TOKEN_RESULT_SOME && token.token_type == TOKEN_TYPE_RPAREN)
    return SQLITE_OK;

  while (1) {
    if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_IDENTIFIER)
      return SQLITE_ERROR;
    const char *key = token.start;
    int keyLength = token.end - token.start;
    rc = vec0_scanner_next(scanner, &token);
    if (rc != VEC0_TOKEN_RESULT_SOME || token.token_type != TOKEN_TYPE_EQ)
      return SQLITE_ERROR;
    rc = vec0_scanner_next(scanner, &token);
    if (rc != VEC0_TOKEN_RESULT_SOME) return SQLITE_ERROR;
    const char *val = token.start;
    int valLength = token.end - token.start;
    int isDigit = token.token_type == TOKEN_TYPE_DIGIT;
    if (!isDigit && token.token_type != TOKEN_TYPE_IDENTIFIER) return SQLITE_ERROR;
    int v = isDigit ? atoi(token.start) : 0;

#define HNSW_KEY(name) \
  (keyLength == (int)sizeof(name) - 1 && sqlite3_strnicmp(key, name, keyLength) == 0)
#define HNSW_VAL(name) \
  (valLength == (int)sizeof(name) - 1 && sqlite3_strnicmp(val, name, valLength) == 0)
    if (HNSW_KEY("quantizer")) {
      if (isDigit) return SQLITE_ERROR;
      if (HNSW_VAL("none")) config->quantizer = VEC0_HNSW_QUANTIZER_NONE;
      else if (HNSW_VAL("int8")) config->quantizer = VEC0_HNSW_QUANTIZER_INT8;
      else return SQLITE_ERROR;
    } else if (!isDigit) {
      return SQLITE_ERROR;
    } else if (HNSW_KEY("m")) {
      if (v < 2 || v > VEC0_HNSW_MAX_M) return SQLITE_ERROR;
      config->m = v;
    } else if (HNSW_KEY("ef_construction")) {
      if (v < 1) return SQLITE_ERROR;
      config->ef_construction = v;
    } else if (HNSW_KEY("ef_search")) {
      if (v < 1) return SQLITE_ERROR;
      config->ef_search = v;
    } else if (HNSW_KEY("cache_mb")) {
      config->cache_mb = v;
    } else {
      return SQLITE_ERROR;
    }
#undef HNSW_KEY
#undef HNSW_VAL

    rc = vec0_scanner_next(scanner, &token);
    if (rc != VEC0_TOKEN_RESULT_SOME) return SQLITE_ERROR;
    if (token.token_type == TOKEN_TYPE_RPAREN) break;
    if (token.token_type != TOKEN_TYPE_COMMA) return SQLITE_ERROR;
    rc = vec0_scanner_next(scanner, &token);
  }
  if (config->ef_construction < config->m) config->ef_construction = config->m;
  return SQLITE_OK;
}

// ============================================================================
// Heaps of (distance, node index)
// ============================================================================

struct HnswItem {
  f32 d;
  i64 id;  // dense node index inside the graph code, rowid in results
};

struct HnswHeap {
  struct HnswItem *a;
  int n, cap;
  int max;  // 1 = max-heap (farthest on top), 0 = min-heap
};

static int hnsw_heap_before(const struct HnswHeap *h, int i, int j) {
  return h->max ? h->a[i].d > h->a[j].d : h->a[i].d < h->a[j].d;
}

static int hnsw_heap_push(struct HnswHeap *h, f32 d, i64 id) {
  if (h->n == h->cap) {
    int cap = h->cap ? h->cap * 2 : 64;
    struct HnswItem *a = sqlite3_realloc64(h->a, (i64)cap * sizeof(*a));
    if (!a) return SQLITE_NOMEM;
    h->a = a;
    h->cap = cap;
  }
  int i = h->n++;
  h->a[i].d = d;
  h->a[i].id = id;
  while (i > 0) {
    int parent = (i - 1) / 2;
    if (!hnsw_heap_before(h, i, parent)) break;
    struct HnswItem t = h->a[i]; h->a[i] = h->a[parent]; h->a[parent] = t;
    i = parent;
  }
  return SQLITE_OK;
}

static struct HnswItem hnsw_heap_pop(struct HnswHeap *h) {
  struct HnswItem top = h->a[0];
  h->a[0] = h->a[--h->n];
  int i = 0;
  for (;;) {
    int l = 2 * i + 1, r = l + 1, best = i;
    if (l < h->n && hnsw_heap_before(h, l, best)) best = l;
    if (r < h->n && hnsw_heap_before(h, r, best)) best = r;
    if (best == i) break;
    struct HnswItem t = h->a[i]; h->a[i] = h->a[best]; h->a[best] = t;
    i = best;
  }
  return top;
}

/** Sort a max-heap in place into ascending order (heapsort's second half). */
static void hnsw_heap_sort(struct HnswHeap *h) {
  int n = h->n;
  while (h->n > 1) {
    struct HnswItem top = h->a[0];
    struct HnswItem last = h->a[h->n - 1];
    h->n--;
    h->a[0] = last;
    int i = 0;
    for (;;) {
      int l = 2 * i + 1, r = l + 1, best = i;
      if (l < h->n && hnsw_heap_before(h, l, best)) best = l;
      if (r < h->n && hnsw_heap_before(h, r, best)) best = r;
      if (best == i) break;
      struct HnswItem t = h->a[i]; h->a[i] = h->a[best]; h->a[best] = t;
      i = best;
    }
    h->a[h->n] = top;
  }
  h->n = n;
}

/** Insertion sort, nearest first; for the short lists of neighbor selection. */
static void hnsw_sort_small(struct HnswItem *a, int n) {
  for (int i = 1; i < n; i++) {
    struct HnswItem t = a[i];
    int j = i - 1;
    while (j >= 0 && a[j].d > t.d) { a[j + 1] = a[j]; j--; }
    a[j + 1] = t;
  }
}

static int hnsw_item_cmp(const void *a, const void *b) {
  f32 x = ((const struct HnswItem *)a)->d, y = ((const struct HnswItem *)b)->d;
  return (x > y) - (x < y);
}

// ============================================================================
// Per-connection cache, in hnswlib's layout
// ============================================================================

#define HNSW_CHUNK_BITS 10
#define HNSW_CHUNK (1 << HNSW_CHUNK_BITS)

enum {
  HNSW_F_VEC = 1,     // vector loaded
  HNSW_F_ABSENT = 2,  // no such node (deleted, or a dangling edge)
  HNSW_F_L0 = 4,      // level-0 neighbor list loaded
  HNSW_F_CODE = 8,    // int8 search code computed
};

struct HnswUpper {  // neighbor list above level 0
  int32_t idx;
  int level;
  int len;
  int32_t *ids;
  u8 used;
};

struct HnswCache {
  int m0;           // level-0 list capacity (2*m)
  size_t vecBytes;
  size_t codeBytes; // 0 = no search codes; else 8 + dims (see hnsw_encode)
  // rowid -> idx (open addressing; hval holds idx+1, 0 = empty)
  i64 *hkey;
  int32_t *hval;
  int hcap;
  int n, cap;       // nodes known, capacity (a multiple of HNSW_CHUNK)
  i64 *rowid;       // idx -> rowid
  u8 *flags;
  u32 *visit;       // visited tags
  u32 tag;
  u8 **vchunk;      // vectors, HNSW_CHUNK per chunk
  u8 **qchunk;      // search codes, HNSW_CHUNK per chunk (codeBytes > 0)
  int32_t **echunk; // level-0 lists, HNSW_CHUNK slots of [count, ids...]
  int nchunks;
  struct HnswUpper *up;
  int upcap, upn;
  struct HnswHeap cand;  // search scratch, reused across operations
  // Entry point (highest level, then lowest rowid), mirrored from SQL.
  int entryKnown, entryFound, entryLevel;
  int32_t entryIdx;
  i64 bytes;
  int broken;       // a load failed mid-operation: clear at the next boundary
  u32 dataVersion;
  int haveVersion;
};

static u64 hnsw_mix(u64 x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static void hnsw_cache_reset(struct HnswCache *c) {
  for (int i = 0; i < c->nchunks; i++) {
    sqlite3_free(c->vchunk[i]);
    sqlite3_free(c->echunk[i]);
    if (c->qchunk) sqlite3_free(c->qchunk[i]);
  }
  for (int i = 0; i < c->upcap; i++) {
    if (c->up[i].used) sqlite3_free(c->up[i].ids);
  }
  sqlite3_free(c->vchunk);
  sqlite3_free(c->echunk);
  sqlite3_free(c->qchunk);
  sqlite3_free(c->up);
  sqlite3_free(c->hkey);
  sqlite3_free(c->hval);
  sqlite3_free(c->rowid);
  sqlite3_free(c->flags);
  sqlite3_free(c->visit);
  sqlite3_free(c->cand.a);
  int m0 = c->m0;
  size_t vecBytes = c->vecBytes, codeBytes = c->codeBytes;
  memset(c, 0, sizeof(*c));
  c->m0 = m0;
  c->vecBytes = vecBytes;
  c->codeBytes = codeBytes;
}

/** Forget every cached HNSW node of this table. */
static void hnsw_cache_clear_all(vec0_vtab *p) {
  for (int i = 0; i < VEC0_MAX_VECTOR_COLUMNS; i++) {
    if (p->hnswCache[i]) hnsw_cache_reset(p->hnswCache[i]);
  }
}

static void hnsw_cache_free_all(vec0_vtab *p) {
  hnsw_cache_clear_all(p);
  for (int i = 0; i < VEC0_MAX_VECTOR_COLUMNS; i++) {
    sqlite3_free(p->hnswCache[i]);
    p->hnswCache[i] = NULL;
  }
}

static inline u8 *hnsw_vecp(const struct HnswCache *c, int32_t idx) {
  return c->vchunk[idx >> HNSW_CHUNK_BITS] + (size_t)(idx & (HNSW_CHUNK - 1)) * c->vecBytes;
}

static inline u8 *hnsw_codep(const struct HnswCache *c, int32_t idx) {
  return c->qchunk[idx >> HNSW_CHUNK_BITS] + (size_t)(idx & (HNSW_CHUNK - 1)) * c->codeBytes;
}

static inline int32_t *hnsw_slot(const struct HnswCache *c, int32_t idx) {
  return c->echunk[idx >> HNSW_CHUNK_BITS] + (size_t)(idx & (HNSW_CHUNK - 1)) * (c->m0 + 1);
}

static int hnsw_grow(struct HnswCache *c) {
  int cap = c->cap + HNSW_CHUNK;
  i64 *rowid = sqlite3_realloc64(c->rowid, (i64)cap * sizeof(i64));
  if (!rowid) return SQLITE_NOMEM;
  c->rowid = rowid;
  u8 *flags = sqlite3_realloc64(c->flags, cap);
  if (!flags) return SQLITE_NOMEM;
  c->flags = flags;
  u32 *visit = sqlite3_realloc64(c->visit, (i64)cap * sizeof(u32));
  if (!visit) return SQLITE_NOMEM;
  c->visit = visit;
  memset(c->visit + c->cap, 0, HNSW_CHUNK * sizeof(u32));
  u8 **vchunk = sqlite3_realloc64(c->vchunk, (i64)(c->nchunks + 1) * sizeof(u8 *));
  if (!vchunk) return SQLITE_NOMEM;
  c->vchunk = vchunk;
  int32_t **echunk = sqlite3_realloc64(c->echunk, (i64)(c->nchunks + 1) * sizeof(int32_t *));
  if (!echunk) return SQLITE_NOMEM;
  c->echunk = echunk;
  u8 *q = NULL;
  if (c->codeBytes) {
    u8 **qchunk = sqlite3_realloc64(c->qchunk, (i64)(c->nchunks + 1) * sizeof(u8 *));
    if (!qchunk) return SQLITE_NOMEM;
    c->qchunk = qchunk;
    q = sqlite3_malloc64((i64)HNSW_CHUNK * c->codeBytes);
    if (!q) return SQLITE_NOMEM;
  }
  u8 *v = sqlite3_malloc64((i64)HNSW_CHUNK * c->vecBytes);
  int32_t *e = sqlite3_malloc64((i64)HNSW_CHUNK * (c->m0 + 1) * sizeof(int32_t));
  if (!v || !e) {
    sqlite3_free(q);
    sqlite3_free(v);
    sqlite3_free(e);
    return SQLITE_NOMEM;
  }
  if (c->codeBytes) c->qchunk[c->nchunks] = q;
  c->vchunk[c->nchunks] = v;
  c->echunk[c->nchunks] = e;
  c->nchunks++;
  c->cap = cap;
  c->bytes += (i64)HNSW_CHUNK *
              ((i64)c->vecBytes + (i64)c->codeBytes + (c->m0 + 1) * 4 + 8 + 1 + 4 + 12);
  return SQLITE_OK;
}

/** Dense index for rowid, allocating an unloaded placeholder if new. */
static int hnsw_idx(struct HnswCache *c, i64 rowid, int32_t *out) {
  if ((c->n + 1) * 10 >= c->hcap * 7) {
    int hcap = c->hcap ? c->hcap * 2 : 1024;
    i64 *hkey = sqlite3_malloc64((i64)hcap * sizeof(i64));
    int32_t *hval = sqlite3_malloc64((i64)hcap * sizeof(int32_t));
    if (!hkey || !hval) {
      sqlite3_free(hkey);
      sqlite3_free(hval);
      return SQLITE_NOMEM;
    }
    memset(hval, 0, (size_t)hcap * sizeof(int32_t));
    u64 mask = (u64)hcap - 1;
    for (int32_t i = 0; i < c->n; i++) {
      u64 s = hnsw_mix((u64)c->rowid[i]) & mask;
      while (hval[s]) s = (s + 1) & mask;
      hkey[s] = c->rowid[i];
      hval[s] = i + 1;
    }
    sqlite3_free(c->hkey);
    sqlite3_free(c->hval);
    c->hkey = hkey;
    c->hval = hval;
    c->bytes += (i64)(hcap - c->hcap) * 12;
    c->hcap = hcap;
  }
  u64 mask = (u64)c->hcap - 1;
  u64 s = hnsw_mix((u64)rowid) & mask;
  while (c->hval[s]) {
    if (c->hkey[s] == rowid) {
      *out = c->hval[s] - 1;
      return SQLITE_OK;
    }
    s = (s + 1) & mask;
  }
  if (c->n == c->cap) {
    int rc = hnsw_grow(c);
    if (rc != SQLITE_OK) return rc;
  }
  int32_t idx = c->n++;
  c->rowid[idx] = rowid;
  c->flags[idx] = 0;
  c->hkey[s] = rowid;
  c->hval[s] = idx + 1;
  *out = idx;
  return SQLITE_OK;
}

static struct HnswUpper *hnsw_upper(struct HnswCache *c, int32_t idx, int level, int create) {
  if (create && (c->upn + 1) * 10 >= c->upcap * 7) {
    int cap = c->upcap ? c->upcap * 2 : 256;
    struct HnswUpper *up = sqlite3_malloc64((i64)cap * sizeof(*up));
    if (!up) return NULL;
    memset(up, 0, (size_t)cap * sizeof(*up));
    for (int i = 0; i < c->upcap; i++) {
      if (!c->up[i].used) continue;
      u64 s = hnsw_mix((u64)c->up[i].idx * 31 + (u64)c->up[i].level) & ((u64)cap - 1);
      while (up[s].used) s = (s + 1) & ((u64)cap - 1);
      up[s] = c->up[i];
    }
    sqlite3_free(c->up);
    c->up = up;
    c->upcap = cap;
  }
  if (!c->upcap) return NULL;
  u64 mask = (u64)c->upcap - 1;
  u64 s = hnsw_mix((u64)idx * 31 + (u64)level) & mask;
  while (c->up[s].used) {
    if (c->up[s].idx == idx && c->up[s].level == level) return &c->up[s];
    s = (s + 1) & mask;
  }
  if (!create) return NULL;
  memset(&c->up[s], 0, sizeof(c->up[s]));
  c->up[s].used = 1;
  c->up[s].idx = idx;
  c->up[s].level = level;
  c->upn++;
  return &c->up[s];
}

struct HnswCtx {
  vec0_vtab *p;
  int col;
  int m;
  size_t dims;
  enum VectorElementType type;
  enum Vec0DistanceMetrics metric;
  vec0_distance_fn dist;  // resolved once per operation
  int defer;              // rebuild: graph writes go to the cache only
  struct HnswCache *c;
};

static int hnsw_data_version(vec0_vtab *p, u32 *v) {
  return sqlite3_file_control(p->db, p->schemaName, SQLITE_FCNTL_DATA_VERSION, v);
}

/** Our own commit moved the data version; the cache already mirrors it. */
static void hnsw_cache_after_commit(vec0_vtab *p) {
  for (int i = 0; i < VEC0_MAX_VECTOR_COLUMNS; i++) {
    struct HnswCache *c = p->hnswCache[i];
    u32 v;
    if (c && c->haveVersion && hnsw_data_version(p, &v) == SQLITE_OK) c->dataVersion = v;
  }
}

/**
 * Start an operation: drop the cache if another connection committed since
 * our last operation, if a previous load failed, or if it is over budget.
 * Never cleared mid-operation, so pointers into it stay valid until hnsw_end().
 */
static int hnsw_begin(struct HnswCtx *x, vec0_vtab *p, int col) {
  struct VectorColumnDefinition *def = &p->vector_columns[col];
  memset(x, 0, sizeof(*x));
  x->p = p;
  x->col = col;
  x->m = def->hnsw.m;
  x->dims = def->dimensions;
  x->type = def->element_type;
  x->metric = def->distance_metric;
  x->dist = vec0_distance_resolve(x->type, x->metric);
  if (!p->hnswCache[col]) {
    struct HnswCache *c = sqlite3_malloc(sizeof(*c));
    if (!c) return SQLITE_NOMEM;
    memset(c, 0, sizeof(*c));
    c->m0 = 2 * def->hnsw.m;
    c->vecBytes = vector_column_byte_size(*def);
    if (def->hnsw.quantizer == VEC0_HNSW_QUANTIZER_INT8 &&
        def->element_type == SQLITE_VEC_ELEMENT_TYPE_FLOAT32 &&
        def->distance_metric != VEC0_DISTANCE_METRIC_L1)
      c->codeBytes = 8 + def->dimensions;
    p->hnswCache[col] = c;
  }
  x->c = p->hnswCache[col];
  u32 v = 0;
  int known = hnsw_data_version(p, &v) == SQLITE_OK;
  if (!known || !x->c->haveVersion || v != x->c->dataVersion || x->c->broken ||
      x->c->bytes > (i64)def->hnsw.cache_mb * 1024 * 1024) {
    hnsw_cache_reset(x->c);
  }
  return SQLITE_OK;
}

/** Finish an operation; the cache stays valid until the data version moves. */
static void hnsw_end(struct HnswCtx *x) {
  u32 v;
  int keep = !x->c->broken && x->p->vector_columns[x->col].hnsw.cache_mb > 0 &&
             hnsw_data_version(x->p, &v) == SQLITE_OK;
  if (!keep) {
    hnsw_cache_reset(x->c);
    return;
  }
  x->c->haveVersion = 1;
  x->c->dataVersion = v;
}

static int hnsw_m_max(struct HnswCtx *x, int level) {
  return level == 0 ? 2 * x->m : x->m;
}

// ============================================================================
// Storage (reads go through the cache; writes update it on success)
// ============================================================================

static void hnsw_encode(const f32 *v, size_t dims, u8 *out);

/** Encode a just-cached vector, so searches never touch the float copy. */
static void hnsw_cache_code(struct HnswCtx *x, int32_t idx) {
  struct HnswCache *c = x->c;
  if (!c->codeBytes) return;
  hnsw_encode((const f32 *)hnsw_vecp(c, idx), x->dims, hnsw_codep(c, idx));
  c->flags[idx] |= HNSW_F_CODE;
}

/** The node's vector, or NULL when the node does not exist. Stable pointer. */
static int hnsw_vector(struct HnswCtx *x, int32_t idx, const void **out) {
  struct HnswCache *c = x->c;
  u8 f = c->flags[idx];
  if (f & HNSW_F_ABSENT) { *out = NULL; return SQLITE_OK; }
  if (f & HNSW_F_VEC) { *out = hnsw_vecp(c, idx); return SQLITE_OK; }
  vec0_vtab *p = x->p;
  int rc = vec0_ensure_stmt(p, &p->stmtHnswNodeRead[x->col],
      "SELECT vector FROM " VEC0_SHADOW_HNSW_NODES_NAME " WHERE rowid = ?", x->col);
  if (rc == SQLITE_OK) {
    sqlite3_stmt *s = p->stmtHnswNodeRead[x->col];
    sqlite3_reset(s);
    sqlite3_bind_int64(s, 1, c->rowid[idx]);
    rc = sqlite3_step(s);
    if (rc == SQLITE_ROW) {
      if ((size_t)sqlite3_column_bytes(s, 0) != c->vecBytes) {
        rc = SQLITE_CORRUPT;
      } else {
        memcpy(hnsw_vecp(c, idx), sqlite3_column_blob(s, 0), c->vecBytes);
        c->flags[idx] |= HNSW_F_VEC;
        hnsw_cache_code(x, idx);
        rc = SQLITE_OK;
      }
    } else if (rc == SQLITE_DONE) {
      c->flags[idx] |= HNSW_F_ABSENT;
      rc = SQLITE_OK;
    }
    sqlite3_reset(s);
  }
  if (rc != SQLITE_OK) {
    c->broken = 1;
    return rc;
  }
  *out = (c->flags[idx] & HNSW_F_ABSENT) ? NULL : hnsw_vecp(c, idx);
  return SQLITE_OK;
}

/** Distance from `q` to node idx; FLT_MAX when the node is gone. */
static int hnsw_distance(struct HnswCtx *x, const void *q, int32_t idx, f32 *d) {
  const void *v;
  int rc = hnsw_vector(x, idx, &v);
  if (rc != SQLITE_OK) return rc;
  *d = v ? x->dist(q, v, &x->dims) : FLT_MAX;
  return SQLITE_OK;
}

/**
 * int8 search code: [f32 scale][f32 squared norm][i8 * dims], with the scale
 * chosen per vector (max |x| / 127) so no global range is needed. The norm is
 * of the dequantized vector, so code-to-code distances are self-consistent.
 */
static void hnsw_encode(const f32 *v, size_t dims, u8 *out) {
  f32 maxabs = 0;
  for (size_t i = 0; i < dims; i++) {
    f32 a = fabsf(v[i]);
    if (a > maxabs) maxabs = a;
  }
  f32 scale = maxabs > 0 ? maxabs / 127.0f : 1.0f;
  i8 *q = (i8 *)(out + 8);
  i32 sq = 0;
  for (size_t i = 0; i < dims; i++) {
    i32 c = (i32)lrintf(v[i] / scale);
    if (c > 127) c = 127;
    if (c < -127) c = -127;
    q[i] = (i8)c;
    sq += c * c;
  }
  f32 norm2 = (f32)sq * scale * scale;
  memcpy(out, &scale, 4);
  memcpy(out + 4, &norm2, 4);
}

static i32 hnsw_dot_i8(const i8 *a, const i8 *b, size_t n) {
  size_t i = 0;
  i32 s = 0;
#if defined(SQLITE_VEC_ENABLE_NEON) && defined(__ARM_FEATURE_DOTPROD)
  int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
  for (; i + 32 <= n; i += 32) {
    acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
    acc1 = vdotq_s32(acc1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
  }
  for (; i + 16 <= n; i += 16) acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
  s = vaddvq_s32(vaddq_s32(acc0, acc1));
#endif
  for (; i < n; i++) s += (i32)a[i] * (i32)b[i];
  return s;
}

/** Approximate distance between two codes, in the column's metric. */
static f32 hnsw_code_distance(const u8 *a, const u8 *b, size_t dims,
                              enum Vec0DistanceMetrics metric) {
  f32 sa, na, sb, nb;
  memcpy(&sa, a, 4); memcpy(&na, a + 4, 4);
  memcpy(&sb, b, 4); memcpy(&nb, b + 4, 4);
  f32 dot = sa * sb * (f32)hnsw_dot_i8((const i8 *)(a + 8), (const i8 *)(b + 8), dims);
  switch (metric) {
  case VEC0_DISTANCE_METRIC_COSINE:
    return (na > 0 && nb > 0) ? 1.0f - dot / sqrtf(na * nb) : 1.0f;
  case VEC0_DISTANCE_METRIC_DOT:
    return -dot;
  default: {
    f32 d = na + nb - 2.0f * dot;
    return d > 0 ? sqrtf(d) : 0.0f;  // vec0 reports L2, not its square
  }
  }
}

/** The node's search code, or NULL when the node does not exist. */
static int hnsw_code(struct HnswCtx *x, int32_t idx, const u8 **out) {
  struct HnswCache *c = x->c;
  if (c->flags[idx] & HNSW_F_CODE) { *out = hnsw_codep(c, idx); return SQLITE_OK; }
  const void *v;
  int rc = hnsw_vector(x, idx, &v);
  if (rc != SQLITE_OK || !v) { *out = NULL; return rc; }
  hnsw_encode((const f32 *)v, x->dims, hnsw_codep(c, idx));
  c->flags[idx] |= HNSW_F_CODE;
  *out = hnsw_codep(c, idx);
  return SQLITE_OK;
}

/**
 * Distance used while searching: exact on vectors, or approximate on codes
 * when `qcode` is set (queries on an int8-quantized column).
 */
static int hnsw_search_distance(struct HnswCtx *x, const void *q, const u8 *qcode,
                                int32_t idx, f32 *d) {
  if (!qcode) return hnsw_distance(x, q, idx, d);
  const u8 *code;
  int rc = hnsw_code(x, idx, &code);
  if (rc != SQLITE_OK) return rc;
  *d = code ? hnsw_code_distance(qcode, code, x->dims, x->metric) : FLT_MAX;
  return SQLITE_OK;
}

/**
 * Neighbors of idx at level, as dense indexes. *ids is borrowed: valid until
 * the next write of that list or the end of the operation.
 */
static int hnsw_edges_read(struct HnswCtx *x, int32_t idx, int level,
                           const int32_t **ids, int *n) {
  struct HnswCache *c = x->c;
  if (level == 0 && (c->flags[idx] & HNSW_F_L0)) {
    int32_t *slot = hnsw_slot(c, idx);
    *ids = slot + 1;
    *n = slot[0];
    return SQLITE_OK;
  }
  if (level > 0) {
    struct HnswUpper *u = hnsw_upper(c, idx, level, 0);
    if (u) {
      *ids = u->ids;
      *n = u->len;
      return SQLITE_OK;
    }
  }

  vec0_vtab *p = x->p;
  int rc = vec0_ensure_stmt(p, &p->stmtHnswEdgesRead[x->col],
      "SELECT neighbors FROM " VEC0_SHADOW_HNSW_EDGES_NAME
      " WHERE rowid = ? AND level = ?", x->col);
  i64 *rowids = NULL;
  int count = 0;
  if (rc == SQLITE_OK) {
    sqlite3_stmt *s = p->stmtHnswEdgesRead[x->col];
    sqlite3_reset(s);
    sqlite3_bind_int64(s, 1, c->rowid[idx]);
    sqlite3_bind_int(s, 2, level);
    rc = sqlite3_step(s);
    if (rc == SQLITE_ROW) {
      int bytes = sqlite3_column_bytes(s, 0);
      count = bytes / (int)sizeof(i64);
      if (bytes % (int)sizeof(i64)) {
        rc = SQLITE_CORRUPT;
      } else if (count > 0 && !(rowids = sqlite3_malloc(bytes))) {
        rc = SQLITE_NOMEM;
      } else {
        if (count > 0) memcpy(rowids, sqlite3_column_blob(s, 0), bytes);
        rc = SQLITE_OK;
      }
    } else if (rc == SQLITE_DONE) {
      rc = SQLITE_OK;
    }
    sqlite3_reset(s);
  }
  int cap = hnsw_m_max(x, level);
  if (count > cap) count = cap;  // a longer list is corrupt; keep the head

  int32_t *dst = NULL;
  if (rc == SQLITE_OK) {
    if (level == 0) {
      dst = hnsw_slot(c, idx) + 1;
    } else {
      struct HnswUpper *u = hnsw_upper(c, idx, level, 1);
      dst = count > 0 ? sqlite3_malloc64((i64)count * sizeof(int32_t)) : NULL;
      if (!u || (count > 0 && !dst)) {
        sqlite3_free(dst);
        rc = SQLITE_NOMEM;
      } else {
        u->ids = dst;
        u->len = count;
        c->bytes += (i64)count * 4 + (i64)sizeof(*u);
      }
    }
  }
  for (int i = 0; rc == SQLITE_OK && i < count; i++) {
    int32_t j;
    rc = hnsw_idx(c, rowids[i], &j);
    if (rc == SQLITE_OK) dst[i] = j;
  }
  sqlite3_free(rowids);
  if (rc != SQLITE_OK) {
    c->broken = 1;
    return rc;
  }
  if (level == 0) {
    hnsw_slot(c, idx)[0] = count;
    c->flags[idx] |= HNSW_F_L0;
  }
  *ids = dst;
  *n = count;
  return SQLITE_OK;
}

/** Write idx's list at level (n <= hnsw_m_max), to SQLite and the cache. */
static int hnsw_edges_write(struct HnswCtx *x, int32_t idx, int level, const int32_t *ids, int n) {
  struct HnswCache *c = x->c;
  vec0_vtab *p = x->p;
  if (x->defer) goto cache_only;
  i64 rowids[2 * VEC0_HNSW_MAX_M];
  for (int i = 0; i < n; i++) rowids[i] = c->rowid[ids[i]];
  int rc = vec0_ensure_stmt(p, &p->stmtHnswEdgesWrite[x->col],
      "INSERT OR REPLACE INTO " VEC0_SHADOW_HNSW_EDGES_NAME
      " (rowid, level, neighbors) VALUES (?, ?, ?)", x->col);
  if (rc != SQLITE_OK) return rc;
  sqlite3_stmt *s = p->stmtHnswEdgesWrite[x->col];
  sqlite3_reset(s);
  sqlite3_bind_int64(s, 1, c->rowid[idx]);
  sqlite3_bind_int(s, 2, level);
  sqlite3_bind_blob(s, 3, rowids, n * (int)sizeof(i64), SQLITE_STATIC);
  rc = sqlite3_step(s);
  sqlite3_reset(s);
  if (rc != SQLITE_DONE) return rc;

cache_only:
  if (level == 0) {
    int32_t *slot = hnsw_slot(c, idx);
    memmove(slot + 1, ids, (size_t)n * sizeof(int32_t));
    slot[0] = n;
    c->flags[idx] |= HNSW_F_L0;
    return SQLITE_OK;
  }
  struct HnswUpper *u = hnsw_upper(c, idx, level, 1);
  int32_t *copy = n > 0 ? sqlite3_malloc64((i64)n * sizeof(int32_t)) : NULL;
  if (!u || (n > 0 && !copy)) {
    sqlite3_free(copy);
    c->broken = 1;  // the cache no longer mirrors the table
    return SQLITE_NOMEM;
  }
  if (n > 0) memcpy(copy, ids, (size_t)n * sizeof(int32_t));
  c->bytes += (i64)(n - u->len) * 4;
  sqlite3_free(u->ids);
  u->ids = copy;
  u->len = n;
  return SQLITE_OK;
}

/** Entry point: the highest-level node. *found = 0 when the graph is empty. */
static int hnsw_entry(struct HnswCtx *x, int32_t *idx, int *level, int *found) {
  struct HnswCache *c = x->c;
  if (c->entryKnown) {
    *found = c->entryFound;
    *idx = c->entryIdx;
    *level = c->entryLevel;
    return SQLITE_OK;
  }
  vec0_vtab *p = x->p;
  int rc = vec0_ensure_stmt(p, &p->stmtHnswEntry[x->col],
      "SELECT rowid, level FROM " VEC0_SHADOW_HNSW_NODES_NAME
      " ORDER BY level DESC, rowid LIMIT 1", x->col);
  if (rc != SQLITE_OK) return rc;
  sqlite3_stmt *s = p->stmtHnswEntry[x->col];
  sqlite3_reset(s);
  rc = sqlite3_step(s);
  *found = rc == SQLITE_ROW;
  i64 rowid = 0;
  if (*found) {
    rowid = sqlite3_column_int64(s, 0);
    *level = sqlite3_column_int(s, 1);
  }
  sqlite3_reset(s);
  if (rc != SQLITE_ROW && rc != SQLITE_DONE) return rc;
  if (*found) {
    rc = hnsw_idx(c, rowid, idx);
    if (rc != SQLITE_OK) return rc;
  }
  c->entryKnown = 1;
  c->entryFound = *found;
  c->entryIdx = *found ? *idx : 0;
  c->entryLevel = *found ? *level : 0;
  return SQLITE_OK;
}

static int hnsw_exec(vec0_vtab *p, const char *fmt, int col) {
  char *zSql = sqlite3_mprintf(fmt, p->schemaName, p->tableName, col);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_exec(p->db, zSql, NULL, NULL, NULL);
  sqlite3_free(zSql);
  return rc;
}

static int hnsw_create_shadow_tables(vec0_vtab *p, int col) {
  // UNIQUE(level, rowid) gives the entry-point index a name that ALTER TABLE
  // RENAME carries along (a CREATE INDEX would keep the old table's name).
  int rc = hnsw_exec(p,
      "CREATE TABLE " VEC0_SHADOW_HNSW_NODES_NAME
      " (rowid INTEGER PRIMARY KEY, level INTEGER NOT NULL, vector BLOB NOT NULL,"
      "  UNIQUE (level, rowid))", col);
  if (rc != SQLITE_OK) return rc;
  return hnsw_exec(p,
      "CREATE TABLE " VEC0_SHADOW_HNSW_EDGES_NAME
      " (rowid INTEGER NOT NULL, level INTEGER NOT NULL, neighbors BLOB NOT NULL,"
      "  PRIMARY KEY (rowid, level)) WITHOUT ROWID", col);
}

static int hnsw_drop_shadow_tables(vec0_vtab *p, int col) {
  int rc = hnsw_exec(p, "DROP TABLE IF EXISTS " VEC0_SHADOW_HNSW_NODES_NAME, col);
  if (rc != SQLITE_OK) return rc;
  return hnsw_exec(p, "DROP TABLE IF EXISTS " VEC0_SHADOW_HNSW_EDGES_NAME, col);
}

// ============================================================================
// Graph algorithms (on dense indexes)
// ============================================================================

static int hnsw_allowed(struct HnswCtx *x, const struct Array *a, int32_t idx) {
  i64 rowid = x->c->rowid[idx];
  return !a || bsearch(&rowid, a->z, a->length, sizeof(i64), _cmp) != NULL;
}

static void hnsw_visit_reset(struct HnswCache *c) {
  if (++c->tag == 0) {
    memset(c->visit, 0, (size_t)c->cap * sizeof(u32));
    c->tag = 1;
  }
}

/**
 * Greedy beam search on one level (Algorithm 2). `eps` seeds the search.
 * Results (at most ef, nearest first) land in *out. With `allow`, only those
 * rowids may enter the results, though every node may be traversed.
 */
static int hnsw_search_level(struct HnswCtx *x, const void *q, const u8 *qcode,
                             const struct HnswItem *eps, int nEps, int ef,
                             int level, const struct Array *allow,
                             struct HnswItem **out, int *nOut) {
  struct HnswCache *c = x->c;
  struct HnswHeap *candp = &c->cand, res = {0};
  candp->n = 0;
  candp->max = 0;
#define cand (*candp)
  res.max = 1;
  int rc = SQLITE_OK;
  hnsw_visit_reset(c);

  for (int i = 0; i < nEps && rc == SQLITE_OK; i++) {
    int32_t e = (int32_t)eps[i].id;
    if (c->visit[e] == c->tag || eps[i].d == FLT_MAX) continue;
    c->visit[e] = c->tag;
    rc = hnsw_heap_push(&cand, eps[i].d, e);
    if (rc == SQLITE_OK && hnsw_allowed(x, allow, e)) rc = hnsw_heap_push(&res, eps[i].d, e);
  }

  while (rc == SQLITE_OK && cand.n > 0) {
    struct HnswItem cur = hnsw_heap_pop(&cand);
    if (res.n >= ef && cur.d > res.a[0].d) break;
    const int32_t *nbrs;
    int n;
    rc = hnsw_edges_read(x, (int32_t)cur.id, level, &nbrs, &n);
    // ponytail: no software prefetch; measured 2-3% slower on Apple M-series
    // (the hardware prefetcher covers this). Re-measure on x86.
    for (int i = 0; rc == SQLITE_OK && i < n; i++) {
      int32_t nb = nbrs[i];
      if (c->visit[nb] == c->tag) continue;
      c->visit[nb] = c->tag;
      f32 d;
      rc = hnsw_search_distance(x, q, qcode, nb, &d);
      if (rc != SQLITE_OK || d == FLT_MAX) continue;
      if (res.n < ef || d < res.a[0].d) {
        rc = hnsw_heap_push(&cand, d, nb);
        if (rc == SQLITE_OK && hnsw_allowed(x, allow, nb)) {
          rc = hnsw_heap_push(&res, d, nb);
          if (rc == SQLITE_OK && res.n > ef) hnsw_heap_pop(&res);
        }
      }
    }
  }

#undef cand
  if (rc != SQLITE_OK) {
    sqlite3_free(res.a);
    return rc;
  }
  hnsw_heap_sort(&res);
  *out = res.a;
  *nOut = res.n;
  return SQLITE_OK;
}

/**
 * Neighbor selection heuristic (Algorithm 4, as in hnswlib): walking the
 * candidates nearest-first, keep one only if it is closer to the base than to
 * every neighbor already kept. Spreads edges across directions.
 * `cands` must be sorted nearest-first. Writes up to `m` indexes to `out`.
 */
static int hnsw_select(struct HnswCtx *x, const struct HnswItem *cands, int n,
                       int m, int32_t *out, int *nOut) {
  int kept = 0, rc = SQLITE_OK;
  const void *keptVec[2 * VEC0_HNSW_MAX_M];
  for (int i = 0; i < n && kept < m; i++) {
    const void *v;
    rc = hnsw_vector(x, (int32_t)cands[i].id, &v);
    if (rc != SQLITE_OK) break;
    if (!v) continue;
    int good = 1;
    for (int j = 0; j < kept && good; j++) {
      if (x->dist(v, keptVec[j], &x->dims) < cands[i].d)
        good = 0;
    }
    if (good) {
      keptVec[kept] = v;
      out[kept++] = (int32_t)cands[i].id;
    }
  }
  *nOut = kept;
  return rc;
}

/**
 * Re-select `node`'s neighbors at `level` from `ids` (which may contain
 * duplicates, the node itself, or `exclude`), then write them. `ids` must not
 * be the cached list of `node` itself.
 */
static int hnsw_reselect(struct HnswCtx *x, int32_t node, int level, const int32_t *ids,
                         int n, int32_t exclude) {
  const void *base;
  int rc = hnsw_vector(x, node, &base);
  if (rc != SQLITE_OK || !base) return rc;
  struct HnswItem *items = sqlite3_malloc64((i64)(n > 0 ? n : 1) * sizeof(*items));
  if (!items) return SQLITE_NOMEM;
  int m = 0;
  for (int i = 0; i < n && rc == SQLITE_OK; i++) {
    if (ids[i] == node || ids[i] == exclude) continue;
    int dup = 0;
    for (int j = 0; j < m && !dup; j++) dup = items[j].id == ids[i];
    if (dup) continue;
    f32 d;
    rc = hnsw_distance(x, base, ids[i], &d);
    if (rc == SQLITE_OK && d != FLT_MAX) { items[m].d = d; items[m].id = ids[i]; m++; }
  }
  int32_t chosen[2 * VEC0_HNSW_MAX_M];
  int nChosen = 0;
  if (rc == SQLITE_OK) {
    hnsw_sort_small(items, m);
    rc = hnsw_select(x, items, m, hnsw_m_max(x, level), chosen, &nChosen);
  }
  if (rc == SQLITE_OK) rc = hnsw_edges_write(x, node, level, chosen, nChosen);
  sqlite3_free(items);
  return rc;
}

/** Level drawn from rowid, so a rebuild reproduces the same graph shape. */
static int hnsw_random_level(struct HnswCtx *x, i64 rowid) {
  double u = (double)(hnsw_mix((u64)rowid) >> 11) * (1.0 / 9007199254740992.0);
  if (u <= 0) u = 1e-300;
  int level = (int)(-log(u) / log((double)x->m));
  return level > VEC0_HNSW_MAX_LEVEL ? VEC0_HNSW_MAX_LEVEL : level;
}

/** Greedy descent (ef=1) from the entry point down to level `stop` + 1. */
static int hnsw_descend(struct HnswCtx *x, const void *q, const u8 *qcode, int32_t entry,
                        int entryLevel, int stop, struct HnswItem *ep) {
  ep->id = entry;
  int rc = hnsw_search_distance(x, q, qcode, entry, &ep->d);
  for (int level = entryLevel; rc == SQLITE_OK && level > stop; level--) {
    for (int changed = 1; changed && rc == SQLITE_OK;) {
      changed = 0;
      const int32_t *nbrs;
      int n;
      rc = hnsw_edges_read(x, (int32_t)ep->id, level, &nbrs, &n);
      for (int i = 0; rc == SQLITE_OK && i < n; i++) {
        f32 d;
        int32_t nb = nbrs[i];
        rc = hnsw_search_distance(x, q, qcode, nb, &d);
        if (rc == SQLITE_OK && d < ep->d) { ep->d = d; ep->id = nb; changed = 1; }
      }
    }
  }
  return rc;
}

// ============================================================================
// Insert / delete / query
// ============================================================================

/** Cache a node with empty neighbor lists, as written by an insert. */
static int hnsw_node_cache(struct HnswCtx *x, int32_t idx, int level, const void *vector) {
  struct HnswCache *c = x->c;
  memcpy(hnsw_vecp(c, idx), vector, c->vecBytes);
  hnsw_slot(c, idx)[0] = 0;
  c->flags[idx] = HNSW_F_VEC | HNSW_F_L0;
  hnsw_cache_code(x, idx);
  for (int l = 1; l <= level; l++) {
    struct HnswUpper *u = hnsw_upper(c, idx, l, 1);
    if (!u) { c->broken = 1; return SQLITE_NOMEM; }
    sqlite3_free(u->ids);
    u->ids = NULL;
    u->len = 0;
  }
  return SQLITE_OK;
}

static int hnsw_node_write(struct HnswCtx *x, int32_t idx, int level, const void *vector) {
  struct HnswCache *c = x->c;
  vec0_vtab *p = x->p;
  if (x->defer) return hnsw_node_cache(x, idx, level, vector);  // row already exists
  int rc = vec0_ensure_stmt(p, &p->stmtHnswNodeInsert[x->col],
      "INSERT INTO " VEC0_SHADOW_HNSW_NODES_NAME " (rowid, level, vector) VALUES (?, ?, ?)",
      x->col);
  if (rc != SQLITE_OK) return rc;
  sqlite3_stmt *s = p->stmtHnswNodeInsert[x->col];
  sqlite3_reset(s);
  sqlite3_bind_int64(s, 1, c->rowid[idx]);
  sqlite3_bind_int(s, 2, level);
  sqlite3_bind_blob(s, 3, vector, (int)c->vecBytes, SQLITE_STATIC);
  rc = sqlite3_step(s);
  sqlite3_reset(s);
  if (rc != SQLITE_DONE) return rc;
  return hnsw_node_cache(x, idx, level, vector);
}

static int hnsw_insert_graph(struct HnswCtx *x, i64 rowid, const void *vector) {
  vec0_vtab *p = x->p;
  int rc, found, entryLevel = 0;
  int32_t entry = 0, idx;
  struct HnswItem *w = NULL;

  rc = hnsw_entry(x, &entry, &entryLevel, &found);
  if (rc == SQLITE_OK) rc = hnsw_idx(x->c, rowid, &idx);
  if (rc != SQLITE_OK) return rc;
  int level = hnsw_random_level(x, rowid);
  rc = hnsw_node_write(x, idx, level, vector);
  if (rc != SQLITE_OK) return rc;
  // Same order as the entry SQL: highest level, then lowest rowid.
  struct HnswCache *c = x->c;
  if (!found || level > entryLevel || (level == entryLevel && rowid < c->rowid[entry])) {
    c->entryKnown = 1;
    c->entryFound = 1;
    c->entryIdx = idx;
    c->entryLevel = level;
  }
  if (!found) return SQLITE_OK;  // first node: entry point, no edges

  struct HnswItem ep;
  rc = hnsw_descend(x, vector, NULL, entry, entryLevel, level, &ep);
  int efc = p->vector_columns[x->col].hnsw.ef_construction;
  const struct HnswItem *eps = &ep;
  int nEps = 1;
  int32_t chosen[VEC0_HNSW_MAX_M], grown[2 * VEC0_HNSW_MAX_M + 1];

  for (int l = level < entryLevel ? level : entryLevel; rc == SQLITE_OK && l >= 0; l--) {
    struct HnswItem *next;
    int nNext;
    rc = hnsw_search_level(x, vector, NULL, eps, nEps, efc, l, NULL, &next, &nNext);
    if (rc != SQLITE_OK) break;
    sqlite3_free(w);
    w = next;
    eps = w;
    nEps = nNext;

    int nChosen;
    rc = hnsw_select(x, w, nNext, x->m, chosen, &nChosen);
    if (rc == SQLITE_OK) rc = hnsw_edges_write(x, idx, l, chosen, nChosen);

    // Reverse edges, shrinking any list that overflows.
    for (int i = 0; rc == SQLITE_OK && i < nChosen; i++) {
      const int32_t *nbrs;
      int n;
      rc = hnsw_edges_read(x, chosen[i], l, &nbrs, &n);
      if (rc != SQLITE_OK) break;
      memcpy(grown, nbrs, (size_t)n * sizeof(int32_t));
      grown[n++] = idx;
      rc = n > hnsw_m_max(x, l) ? hnsw_reselect(x, chosen[i], l, grown, n, -1)
                                : hnsw_edges_write(x, chosen[i], l, grown, n);
    }
  }
  sqlite3_free(w);
  return rc;
}

static int hnsw_delete_graph(struct HnswCtx *x, i64 rowid) {
  vec0_vtab *p = x->p;
  struct HnswCache *c = x->c;
  int rc, level = -1;
  int32_t *mine = NULL, *merged = NULL, idx;

  {
    sqlite3_stmt *s = NULL;
    char *zSql = sqlite3_mprintf("SELECT level FROM " VEC0_SHADOW_HNSW_NODES_NAME
                                 " WHERE rowid = ?", p->schemaName, p->tableName, x->col);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &s, NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
    sqlite3_bind_int64(s, 1, rowid);
    if (sqlite3_step(s) == SQLITE_ROW) level = sqlite3_column_int(s, 0);
    rc = sqlite3_finalize(s);
    if (rc != SQLITE_OK) return rc;
  }
  if (level < 0) return SQLITE_OK;
  rc = hnsw_idx(c, rowid, &idx);

  for (int l = 0; rc == SQLITE_OK && l <= level; l++) {
    const int32_t *cached;
    int nMine;
    rc = hnsw_edges_read(x, idx, l, &cached, &nMine);
    if (rc != SQLITE_OK) break;
    // copy: repairing neighbors rewrites cached lists
    sqlite3_free(mine);
    mine = sqlite3_malloc64((i64)(nMine > 0 ? nMine : 1) * sizeof(int32_t));
    if (!mine) { rc = SQLITE_NOMEM; break; }
    memcpy(mine, cached, (size_t)nMine * sizeof(int32_t));
    for (int i = 0; rc == SQLITE_OK && i < nMine; i++) {
      const int32_t *theirs;
      int nTheirs;
      rc = hnsw_edges_read(x, mine[i], l, &theirs, &nTheirs);
      if (rc != SQLITE_OK) break;
      merged = sqlite3_malloc64((i64)(nTheirs + nMine + 1) * sizeof(int32_t));
      if (!merged) { rc = SQLITE_NOMEM; break; }
      memcpy(merged, theirs, (size_t)nTheirs * sizeof(int32_t));
      memcpy(merged + nTheirs, mine, (size_t)nMine * sizeof(int32_t));
      rc = hnsw_reselect(x, mine[i], l, merged, nTheirs + nMine, idx);
      sqlite3_free(merged);
      merged = NULL;
    }
  }
  if (rc == SQLITE_OK)
    rc = vec0_exec_rowid(p, "DELETE FROM " VEC0_SHADOW_HNSW_EDGES_NAME " WHERE rowid = ?", x->col, rowid);
  if (rc == SQLITE_OK)
    rc = vec0_exec_rowid(p, "DELETE FROM " VEC0_SHADOW_HNSW_NODES_NAME " WHERE rowid = ?", x->col, rowid);
  if (rc == SQLITE_OK) {
    // mirror the deletes: absent node, empty lists
    if (c->entryKnown && c->entryFound && c->entryIdx == idx) c->entryKnown = 0;
    c->flags[idx] = HNSW_F_ABSENT | HNSW_F_L0;
    hnsw_slot(c, idx)[0] = 0;
    for (int l = 1; l <= level; l++) {
      struct HnswUpper *u = hnsw_upper(c, idx, l, 0);
      if (u) { sqlite3_free(u->ids); u->ids = NULL; u->len = 0; }
    }
  }
  sqlite3_free(mine);
  sqlite3_free(merged);
  return rc;
}

static int hnsw_insert(vec0_vtab *p, int col, i64 rowid, const void *vector) {
  struct HnswCtx x;
  int rc = hnsw_begin(&x, p, col);
  if (rc != SQLITE_OK) return rc;
  rc = hnsw_insert_graph(&x, rowid, vector);
  if (rc != SQLITE_OK) x.c->broken = 1;  // a partial write may be cached
  hnsw_end(&x);
  return rc;
}

static int hnsw_delete(vec0_vtab *p, int col, i64 rowid) {
  struct HnswCtx x;
  int rc = hnsw_begin(&x, p, col);
  if (rc != SQLITE_OK) return rc;
  rc = hnsw_delete_graph(&x, rowid);
  if (rc != SQLITE_OK) x.c->broken = 1;
  hnsw_end(&x);
  return rc;
}

/** UPDATE of the vector: unlink the node and link it again. */
static int hnsw_update(vec0_vtab *p, int col, i64 rowid, const void *vector) {
  struct HnswCtx x;
  int rc = hnsw_begin(&x, p, col);
  if (rc != SQLITE_OK) return rc;
  rc = hnsw_delete_graph(&x, rowid);
  if (rc == SQLITE_OK) rc = hnsw_insert_graph(&x, rowid, vector);
  if (rc != SQLITE_OK) x.c->broken = 1;
  hnsw_end(&x);
  return rc;
}

static int hnsw_get_vector_data(vec0_vtab *p, i64 rowid, int col,
                                void **outVector, int *outVectorSize) {
  struct HnswCtx x;
  int rc = hnsw_begin(&x, p, col);
  if (rc != SQLITE_OK) return rc;
  int32_t idx;
  const void *v = NULL;
  rc = hnsw_idx(x.c, rowid, &idx);
  if (rc == SQLITE_OK) rc = hnsw_vector(&x, idx, &v);
  if (rc == SQLITE_OK && !v) rc = SQLITE_EMPTY;
  if (rc == SQLITE_OK) {
    *outVector = sqlite3_malloc64(x.c->vecBytes);
    if (!*outVector) rc = SQLITE_NOMEM;
    else memcpy(*outVector, v, x.c->vecBytes);
    if (outVectorSize) *outVectorSize = (int)x.c->vecBytes;
  }
  hnsw_end(&x);
  return rc;
}

static int hnsw_query_knn(vec0_vtab *p, int col, const void *query,
                          const struct Array *rowidsIn, i64 k,
                          struct vec0_query_knn_data *knn) {
  struct HnswCtx x;
  memset(knn, 0, sizeof(*knn));
  knn->k = k;
  int rc = hnsw_begin(&x, p, col);
  if (rc != SQLITE_OK) return rc;
  struct HnswItem *res = NULL;
  int nRes = 0;

  if (rowidsIn && rowidsIn->length <= VEC0_HNSW_EXACT_FILTER_MAX) {
    res = sqlite3_malloc64((i64)(rowidsIn->length ? rowidsIn->length : 1) * sizeof(*res));
    if (!res) rc = SQLITE_NOMEM;
    for (size_t i = 0; rc == SQLITE_OK && i < rowidsIn->length; i++) {
      int32_t idx;
      f32 d;
      rc = hnsw_idx(x.c, ((const i64 *)rowidsIn->z)[i], &idx);
      if (rc == SQLITE_OK) rc = hnsw_distance(&x, query, idx, &d);
      if (rc == SQLITE_OK && d != FLT_MAX) { res[nRes].d = d; res[nRes].id = idx; nRes++; }
    }
    if (rc == SQLITE_OK) qsort(res, nRes, sizeof(*res), hnsw_item_cmp);
  } else {
    int32_t entry;
    int entryLevel, found;
    rc = hnsw_entry(&x, &entry, &entryLevel, &found);
    if (rc == SQLITE_OK && found) {
      u8 qcodeBuf[8 + SQLITE_VEC_VEC0_MAX_DIMENSIONS];
      const u8 *qcode = NULL;
      if (x.c->codeBytes) {
        hnsw_encode((const f32 *)query, x.dims, qcodeBuf);
        qcode = qcodeBuf;
      }
      struct HnswItem ep;
      rc = hnsw_descend(&x, query, qcode, entry, entryLevel, 0, &ep);
      int ef = p->vector_columns[col].hnsw.ef_search;
      if (ef < k) ef = (int)k;
      if (rc == SQLITE_OK)
        rc = hnsw_search_level(&x, query, qcode, &ep, 1, ef, 0, rowidsIn, &res, &nRes);
      // Codes ranked the candidates; exact distances decide the answer.
      for (int i = 0; qcode && rc == SQLITE_OK && i < nRes; i++)
        rc = hnsw_distance(&x, query, (int32_t)res[i].id, &res[i].d);
      if (qcode && rc == SQLITE_OK) hnsw_sort_small(res, nRes);
    }
  }

  if (rc == SQLITE_OK && nRes > 0) {
    i64 n = nRes < k ? nRes : k;
    knn->rowids = sqlite3_malloc64(n * sizeof(i64));
    knn->distances = sqlite3_malloc64(n * sizeof(f32));
    if (!knn->rowids || !knn->distances) {
      sqlite3_free(knn->rowids);
      sqlite3_free(knn->distances);
      knn->rowids = NULL;
      knn->distances = NULL;
      rc = SQLITE_NOMEM;
    } else {
      for (i64 i = 0; i < n; i++) {
        knn->rowids[i] = x.c->rowid[res[i].id];
        knn->distances[i] = res[i].d;
      }
      knn->k_used = n;
    }
  }
  sqlite3_free(res);
  hnsw_end(&x);
  return rc;
}

/**
 * Relink every node from scratch, in rowid order, with all neighbor-list
 * writes held in the cache, then write every list once. One statement, so a
 * failure rolls the whole rebuild back. Needs the whole graph in memory for
 * the duration, whatever cache_mb says.
 */
static int hnsw_rebuild(vec0_vtab *p, int col) {
  struct HnswCtx x;
  int rc = hnsw_begin(&x, p, col);
  if (rc != SQLITE_OK) return rc;
  hnsw_cache_reset(x.c);
  x.c->entryKnown = 1;  // nothing is linked yet, whatever the nodes table says
  x.c->entryFound = 0;
  x.defer = 1;
  int *levels = NULL;
  int nLevels = 0;
  sqlite3_stmt *s = NULL;
  rc = hnsw_exec(p, "DELETE FROM " VEC0_SHADOW_HNSW_EDGES_NAME, col);
  if (rc == SQLITE_OK) {
    char *zSql = sqlite3_mprintf("SELECT rowid, level, vector FROM " VEC0_SHADOW_HNSW_NODES_NAME
                                 " ORDER BY rowid", p->schemaName, p->tableName, col);
    rc = zSql ? sqlite3_prepare_v2(p->db, zSql, -1, &s, NULL) : SQLITE_NOMEM;
    sqlite3_free(zSql);
  }
  while (rc == SQLITE_OK) {
    int step = sqlite3_step(s);
    if (step == SQLITE_DONE) break;
    if (step != SQLITE_ROW) { rc = step; break; }
    if ((size_t)sqlite3_column_bytes(s, 2) != x.c->vecBytes) { rc = SQLITE_CORRUPT; break; }
    i64 rowid = sqlite3_column_int64(s, 0);
    if (sqlite3_column_int(s, 1) != hnsw_random_level(&x, rowid)) { rc = SQLITE_CORRUPT; break; }
    rc = hnsw_insert_graph(&x, rowid, sqlite3_column_blob(s, 2));
  }
  sqlite3_finalize(s);

  // Write every list once. Node levels come back from the cache shape.
  x.defer = 0;
  if (rc == SQLITE_OK) {
    nLevels = x.c->n;
    levels = sqlite3_malloc64((i64)(nLevels ? nLevels : 1) * sizeof(int));
    if (!levels) rc = SQLITE_NOMEM;
  }
  for (int32_t i = 0; rc == SQLITE_OK && i < nLevels; i++) {
    levels[i] = (x.c->flags[i] & HNSW_F_ABSENT) ? -1 : hnsw_random_level(&x, x.c->rowid[i]);
  }
  for (int32_t i = 0; rc == SQLITE_OK && i < nLevels; i++) {
    for (int l = 0; rc == SQLITE_OK && l <= levels[i]; l++) {
      const int32_t *ids;
      int n;
      int32_t copy[2 * VEC0_HNSW_MAX_M];
      rc = hnsw_edges_read(&x, i, l, &ids, &n);
      if (rc == SQLITE_OK) {
        memcpy(copy, ids, (size_t)n * sizeof(int32_t));
        rc = hnsw_edges_write(&x, i, l, copy, n);
      }
    }
  }
  sqlite3_free(levels);
  if (rc != SQLITE_OK) x.c->broken = 1;
  hnsw_end(&x);
  return rc;
}

/**
 * Runtime tuning, e.g. insert into t(t) values ('ef_search=128').
 * Applies to every HNSW column of the table, for this connection.
 */
static int hnsw_handle_command(vec0_vtab *p, const char *command) {
  if (strcmp(command, "hnsw-rebuild") == 0) {
    int handled = 0;
    for (int i = 0; i < p->numVectorColumns; i++) {
      if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_HNSW) continue;
      int rc = hnsw_rebuild(p, i);
      if (rc != SQLITE_OK) return rc;
      handled = 1;
    }
    return handled ? SQLITE_OK : SQLITE_EMPTY;
  }
  const char *eq = strchr(command, '=');
  if (!eq) return SQLITE_EMPTY;
  size_t keyLen = (size_t)(eq - command);
  int is_ef = keyLen == 9 && strncmp(command, "ef_search", 9) == 0;
  int is_cache = keyLen == 8 && strncmp(command, "cache_mb", 8) == 0;
  if (!is_ef && !is_cache) return SQLITE_EMPTY;
  int min = is_ef ? 1 : 0;
  char *end;
  long v = strtol(eq + 1, &end, 10);
  if (end == eq + 1 || *end || v < min || v > INT_MAX) {
    vtab_set_error(&p->base, "%.*s must be an integer >= %d", (int)keyLen, command, min);
    return SQLITE_ERROR;
  }
  int handled = 0;
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].index_type != VEC0_INDEX_TYPE_HNSW) continue;
    if (is_ef) p->vector_columns[i].hnsw.ef_search = (int)v;
    else p->vector_columns[i].hnsw.cache_mb = (int)v;
    handled = 1;
  }
  if (is_cache) hnsw_cache_clear_all(p);
  return handled ? SQLITE_OK : SQLITE_EMPTY;
}

#endif /* SQLITE_VEC_HNSW_C */
