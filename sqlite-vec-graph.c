/**
 * sqlite-vec-graph.c — breadth-first traversal over ordinary SQLite edge tables.
 *
 * #include'd into sqlite-vec.c; graph_register() runs from sqlite3_vec_init().
 *
 *   select node, depth from graph_bfs('edges', 'src', 'dst', :start, :max_depth);
 *   select step, node from graph_shortest_path('edges', 'src', 'dst', :a, :b);
 *
 * Edges are directed src -> dst. Any table or view works, so an undirected
 * graph is a `UNION ALL` view over both directions.
 *
 * Each connection caches a compressed sparse row (CSR) copy of up to
 * GRAPH_CACHE_SLOTS (table, src, dst) graphs, laid out like LadybugDB's
 * adjacency: nodes get dense 32-bit indexes, one offsets array and one
 * neighbor array sized exactly once (no per-batch slack), and a tag array for
 * the visited set. A cached graph is reused only while the main pager's data
 * version (moves on every commit, ours included) and sqlite3_total_changes64()
 * (moves on our own row writes, also to temp tables) are unchanged.
 *
 * Inside a write transaction the graph is rebuilt for every statement and not
 * cached: ROLLBACK TO a savepoint reverts rows without moving either counter.
 */

#ifndef SQLITE_VEC_GRAPH_C
#define SQLITE_VEC_GRAPH_C

#ifndef SQLITE_VEC_H
#include "sqlite-vec.c" // IWYU pragma: keep
#endif

// ponytail: the bound is a slot count, not bytes. A cached graph costs about
// 4 B/edge + 32 B/node (ids, offsets, hash at <= 50% load, tag, parent), plus
// 8 B/edge transient during the build. Eight slots evict least-recently-used;
// add a byte budget (like hnsw cache_mb) if graphs get large.
#define GRAPH_CACHE_SLOTS 8
#define GRAPH_MAX_NODES 0xFFFFFFF0u

typedef struct GraphCsr GraphCsr;
struct GraphCsr {
  char *zKey;    // "table\x1fsrc\x1fdst"
  u32 dataVersion;
  i64 changes;
  u64 lastUse;
  u32 nNode;
  u32 nEdge;
  i64 *ids;      // dense index -> node id
  u32 *off;      // nNode + 1 offsets into adj
  u32 *adj;      // dense neighbor indexes
  u32 *slot;     // open-addressing id hash: dense index + 1, 0 = empty
  u32 mask;      // slot capacity - 1
  u32 *tag;      // visited set: tag[i] == gen
  u32 *parent;   // valid where tag[i] == gen
  u32 gen;
};

/** One per connection, shared by both modules (passed as pAux). */
typedef struct GraphCache GraphCache;
struct GraphCache {
  GraphCsr *a[GRAPH_CACHE_SLOTS];
  u64 clock;
};

static void graph_csr_free(GraphCsr *g) {
  if (!g) return;
  sqlite3_free(g->zKey);
  sqlite3_free(g->ids);
  sqlite3_free(g->off);
  sqlite3_free(g->adj);
  sqlite3_free(g->slot);
  sqlite3_free(g->tag);
  sqlite3_free(g->parent);
  sqlite3_free(g);
}

static void graph_cache_free(void *p) {
  GraphCache *c = p;
  for (int i = 0; i < GRAPH_CACHE_SLOTS; i++) graph_csr_free(c->a[i]);
  sqlite3_free(c);
}

static u32 graph_hash(i64 id) {
  u64 x = (u64)id + 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return (u32)(x ^ (x >> 31));
}

/** Dense index of id, or -1. */
static i64 graph_find(GraphCsr *g, i64 id) {
  if (!g->slot) return -1;
  for (u32 h = graph_hash(id) & g->mask;; h = (h + 1) & g->mask) {
    u32 s = g->slot[h];
    if (s == 0) return -1;
    if (g->ids[s - 1] == id) return s - 1;
  }
}

static int graph_rehash(GraphCsr *g, u32 cap) {
  u32 *slot = sqlite3_malloc64((u64)cap * sizeof(u32));
  if (!slot) return SQLITE_NOMEM;
  memset(slot, 0, (u64)cap * sizeof(u32));
  sqlite3_free(g->slot);
  g->slot = slot;
  g->mask = cap - 1;
  for (u32 i = 0; i < g->nNode; i++) {
    u32 h = graph_hash(g->ids[i]) & g->mask;
    while (slot[h]) h = (h + 1) & g->mask;
    slot[h] = i + 1;
  }
  return SQLITE_OK;
}

/** Dense index of id, adding it if new. */
static int graph_intern(GraphCsr *g, i64 id, u32 *pIdx, u32 *capIds) {
  i64 found = graph_find(g, id);
  if (found >= 0) { *pIdx = (u32)found; return SQLITE_OK; }
  if (g->nNode >= GRAPH_MAX_NODES) return SQLITE_FULL;
  if (g->nNode == *capIds) {
    u32 cap = *capIds ? *capIds * 2 : 1024;
    i64 *ids = sqlite3_realloc64(g->ids, (u64)cap * sizeof(i64));
    if (!ids) return SQLITE_NOMEM;
    g->ids = ids;
    *capIds = cap;
  }
  g->ids[g->nNode++] = id;
  // Keep the hash at most half full.
  if (!g->slot || (u64)g->nNode * 2 > (u64)g->mask + 1) {
    int rc = graph_rehash(g, g->slot ? (g->mask + 1) * 2 : 2048);
    if (rc != SQLITE_OK) return rc;
  } else {
    u32 h = graph_hash(id) & g->mask;
    while (g->slot[h]) h = (h + 1) & g->mask;
    g->slot[h] = g->nNode;
  }
  *pIdx = g->nNode - 1;
  return SQLITE_OK;
}

static int graph_endpoint(sqlite3_stmt *stmt, int col, i64 *out, char **pzErr) {
  int t = sqlite3_column_type(stmt, col);
  if (t == SQLITE_NULL) return SQLITE_DONE;
  if (t != SQLITE_INTEGER) {
    *pzErr = sqlite3_mprintf("graph: edge endpoints must be integers, got %s",
                             t == SQLITE_TEXT ? "TEXT" : t == SQLITE_FLOAT ? "REAL" : "BLOB");
    return SQLITE_ERROR;
  }
  *out = sqlite3_column_int64(stmt, col);
  return SQLITE_OK;
}

/** 1 if zTable (table or view) has column zCol; SQLite would otherwise read a
 * missing double-quoted column as a string literal. */
static int graph_has_column(sqlite3 *db, const char *zTable, const char *zCol) {
  sqlite3_stmt *stmt;
  int found = 0;
  if (sqlite3_prepare_v2(db,
          "SELECT 1 FROM pragma_table_xinfo(?) WHERE name = ? COLLATE NOCASE",
          -1, &stmt, NULL) != SQLITE_OK)
    return 0;
  sqlite3_bind_text(stmt, 1, zTable, -1, SQLITE_STATIC);
  sqlite3_bind_text(stmt, 2, zCol, -1, SQLITE_STATIC);
  found = sqlite3_step(stmt) == SQLITE_ROW;
  sqlite3_finalize(stmt);
  return found;
}

/** Scan the edge table once and build its CSR. Rows with a NULL endpoint are skipped. */
static int graph_build(sqlite3 *db, const char *zTable, const char *zSrc,
                       const char *zDst, GraphCsr **out, char **pzErr) {
  int rc;
  sqlite3_stmt *stmt = NULL;
  u32 *pairs = NULL; // (src, dst) dense pairs, transient
  u64 nPair = 0, capPair = 0;
  u32 capIds = 0;
  GraphCsr *g = sqlite3_malloc(sizeof(*g));
  if (!g) return SQLITE_NOMEM;
  memset(g, 0, sizeof(*g));

  if (!graph_has_column(db, zTable, zSrc) || !graph_has_column(db, zTable, zDst)) {
    *pzErr = sqlite3_mprintf("graph: cannot read edges %s(%s, %s): no such table or column",
                             zTable, zSrc, zDst);
    rc = SQLITE_ERROR;
    goto done;
  }
  // %w quotes identifiers, so the names cannot inject SQL.
  char *zSql = sqlite3_mprintf("SELECT \"%w\", \"%w\" FROM \"%w\"", zSrc, zDst, zTable);
  if (!zSql) { rc = SQLITE_NOMEM; goto done; }
  rc = sqlite3_prepare_v2(db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    *pzErr = sqlite3_mprintf("graph: cannot read edges %s(%s, %s): %s", zTable,
                             zSrc, zDst, sqlite3_errmsg(db));
    goto done;
  }
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    i64 s, d;
    u32 si, di;
    int r1 = graph_endpoint(stmt, 0, &s, pzErr);
    if (r1 == SQLITE_ERROR) { rc = r1; goto done; }
    int r2 = r1 == SQLITE_OK ? graph_endpoint(stmt, 1, &d, pzErr) : SQLITE_DONE;
    if (r2 == SQLITE_ERROR) { rc = r2; goto done; }
    if (r1 != SQLITE_OK || r2 != SQLITE_OK) continue;
    if (nPair >= GRAPH_MAX_NODES) { rc = SQLITE_FULL; goto done; }
    if ((rc = graph_intern(g, s, &si, &capIds)) != SQLITE_OK) goto done;
    if ((rc = graph_intern(g, d, &di, &capIds)) != SQLITE_OK) goto done;
    if (nPair == capPair) {
      capPair = capPair ? capPair * 2 : 4096;
      u32 *p = sqlite3_realloc64(pairs, capPair * 2 * sizeof(u32));
      if (!p) { rc = SQLITE_NOMEM; goto done; }
      pairs = p;
    }
    pairs[2 * nPair] = si;
    pairs[2 * nPair + 1] = di;
    nPair++;
  }
  if (rc != SQLITE_DONE) {
    *pzErr = sqlite3_mprintf("graph: reading edges: %s", sqlite3_errmsg(db));
    goto done;
  }

  // Counting sort of pairs by source into exactly-sized CSR arrays.
  g->nEdge = (u32)nPair;
  g->off = sqlite3_malloc64(((u64)g->nNode + 1) * sizeof(u32));
  g->adj = sqlite3_malloc64((nPair ? nPair : 1) * sizeof(u32));
  g->tag = sqlite3_malloc64(((u64)g->nNode + 1) * sizeof(u32));
  g->parent = sqlite3_malloc64(((u64)g->nNode + 1) * sizeof(u32));
  if (!g->off || !g->adj || !g->tag || !g->parent) { rc = SQLITE_NOMEM; goto done; }
  memset(g->off, 0, ((u64)g->nNode + 1) * sizeof(u32));
  memset(g->tag, 0, ((u64)g->nNode + 1) * sizeof(u32));
  for (u64 i = 0; i < nPair; i++) g->off[pairs[2 * i] + 1]++;
  for (u32 i = 0; i < g->nNode; i++) g->off[i + 1] += g->off[i];
  // Reuse tag as the per-source fill cursor, then clear it.
  for (u64 i = 0; i < nPair; i++) {
    u32 s = pairs[2 * i];
    g->adj[g->off[s] + g->tag[s]++] = pairs[2 * i + 1];
  }
  memset(g->tag, 0, ((u64)g->nNode + 1) * sizeof(u32));
  if (g->nNode && capIds > g->nNode) {
    i64 *ids = sqlite3_realloc64(g->ids, (u64)g->nNode * sizeof(i64));
    if (ids) g->ids = ids;
  }
  rc = SQLITE_OK;

done:
  sqlite3_finalize(stmt);
  sqlite3_free(pairs);
  if (rc == SQLITE_FULL && !*pzErr)
    *pzErr = sqlite3_mprintf("graph: more than %u nodes or edges", GRAPH_MAX_NODES);
  if (rc != SQLITE_OK) { graph_csr_free(g); return rc; }
  *out = g;
  return SQLITE_OK;
}

/**
 * The current graph for (table, src, dst). *pFresh is set when the caller owns
 * the result (write transaction) and must free it after the statement's scan.
 */
static int graph_get(sqlite3 *db, GraphCache *c, const char *zTable,
                     const char *zSrc, const char *zDst, GraphCsr **out,
                     int *pFresh, char **pzErr) {
  *pFresh = 0;
  if (sqlite3_txn_state(db, NULL) == SQLITE_TXN_WRITE) {
    *pFresh = 1;
    return graph_build(db, zTable, zSrc, zDst, out, pzErr);
  }
  u32 dv = 0;
  int known = sqlite3_file_control(db, "main", SQLITE_FCNTL_DATA_VERSION, &dv) == SQLITE_OK;
  i64 changes = sqlite3_total_changes64(db);
  char *zKey = sqlite3_mprintf("%s\x1f%s\x1f%s", zTable, zSrc, zDst);
  if (!zKey) return SQLITE_NOMEM;
  int victim = 0;
  for (int i = 0; i < GRAPH_CACHE_SLOTS; i++) {
    GraphCsr *g = c->a[i];
    if (g && strcmp(g->zKey, zKey) == 0) {
      if (known && g->dataVersion == dv && g->changes == changes) {
        sqlite3_free(zKey);
        g->lastUse = ++c->clock;
        *out = g;
        return SQLITE_OK;
      }
      victim = i;
      break;
    }
    if (!g || (c->a[victim] && g->lastUse < c->a[victim]->lastUse)) victim = i;
  }
  GraphCsr *g;
  int rc = graph_build(db, zTable, zSrc, zDst, &g, pzErr);
  if (rc != SQLITE_OK) { sqlite3_free(zKey); return rc; }
  g->zKey = zKey;
  g->dataVersion = dv;
  // An unknown data version never matches again, so the next call rebuilds.
  g->changes = known ? changes : -1;
  g->lastUse = ++c->clock;
  graph_csr_free(c->a[victim]);
  c->a[victim] = g;
  *out = g;
  return SQLITE_OK;
}

static u32 graph_next_gen(GraphCsr *g) {
  if (++g->gen == 0) {
    memset(g->tag, 0, ((u64)g->nNode + 1) * sizeof(u32));
    g->gen = 1;
  }
  return g->gen;
}

/* ---- virtual table ---- */

enum { GRAPH_BFS, GRAPH_PATH };

// Column layout shared by both modules: two output columns, then hidden args.
#define GRAPH_COL_OUT0 0
#define GRAPH_COL_OUT1 1
#define GRAPH_COL_TABLE 2
#define GRAPH_COL_SRC 3
#define GRAPH_COL_DST 4
#define GRAPH_COL_ARG0 5 // bfs: start, path: source
#define GRAPH_COL_ARG1 6 // bfs: max_depth (optional), path: target
#define GRAPH_NCOL 7

typedef struct graph_vtab graph_vtab;
struct graph_vtab {
  sqlite3_vtab base;
  sqlite3 *db;
  GraphCache *cache;
  int kind;
};

typedef struct graph_cursor graph_cursor;
struct graph_cursor {
  sqlite3_vtab_cursor base;
  i64 *a;  // output column 0 (bfs: node, path: step)
  i64 *b;  // output column 1 (bfs: depth, path: node)
  i64 n;
  i64 i;
};

static int graph_connect_kind(sqlite3 *db, void *pAux, sqlite3_vtab **ppVtab,
                              int kind) {
  int rc = sqlite3_declare_vtab(
      db, kind == GRAPH_BFS
              ? "CREATE TABLE x(node, depth, edge_table hidden, src_column hidden,"
                " dst_column hidden, start hidden, max_depth hidden)"
              : "CREATE TABLE x(step, node, edge_table hidden, src_column hidden,"
                " dst_column hidden, source hidden, target hidden)");
  if (rc != SQLITE_OK) return rc;
  graph_vtab *p = sqlite3_malloc(sizeof(*p));
  if (!p) return SQLITE_NOMEM;
  memset(p, 0, sizeof(*p));
  p->db = db;
  p->cache = pAux;
  p->kind = kind;
  *ppVtab = &p->base;
  return SQLITE_OK;
}

static int graph_bfsConnect(sqlite3 *db, void *pAux, int argc,
                            const char *const *argv, sqlite3_vtab **ppVtab,
                            char **pzErr) {
  UNUSED_PARAMETER(argc); UNUSED_PARAMETER(argv); UNUSED_PARAMETER(pzErr);
  return graph_connect_kind(db, pAux, ppVtab, GRAPH_BFS);
}

static int graph_pathConnect(sqlite3 *db, void *pAux, int argc,
                             const char *const *argv, sqlite3_vtab **ppVtab,
                             char **pzErr) {
  UNUSED_PARAMETER(argc); UNUSED_PARAMETER(argv); UNUSED_PARAMETER(pzErr);
  return graph_connect_kind(db, pAux, ppVtab, GRAPH_PATH);
}

static int graphDisconnect(sqlite3_vtab *pVtab) {
  sqlite3_free(pVtab);
  return SQLITE_OK;
}

static int graphOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor) {
  UNUSED_PARAMETER(p);
  graph_cursor *cur = sqlite3_malloc(sizeof(*cur));
  if (!cur) return SQLITE_NOMEM;
  memset(cur, 0, sizeof(*cur));
  *ppCursor = &cur->base;
  return SQLITE_OK;
}

static int graphClose(sqlite3_vtab_cursor *cur) {
  graph_cursor *c = (graph_cursor *)cur;
  sqlite3_free(c->a);
  sqlite3_free(c->b);
  sqlite3_free(c);
  return SQLITE_OK;
}

/** idxNum is a bitmask of the hidden columns bound, in column order in argv. */
static int graphBestIndex(sqlite3_vtab *pVTab, sqlite3_index_info *info) {
  graph_vtab *p = (graph_vtab *)pVTab;
  int seen[GRAPH_NCOL] = {0};
  int mask = 0;
  for (int i = 0; i < info->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &info->aConstraint[i];
    if (c->iColumn < GRAPH_COL_TABLE || c->op != SQLITE_INDEX_CONSTRAINT_EQ) continue;
    if (!c->usable) return SQLITE_CONSTRAINT;
    if (seen[c->iColumn]) continue;
    seen[c->iColumn] = i + 1;
    mask |= 1 << c->iColumn;
  }
  int required = (1 << GRAPH_COL_TABLE) | (1 << GRAPH_COL_SRC) |
                 (1 << GRAPH_COL_DST) | (1 << GRAPH_COL_ARG0);
  if (p->kind == GRAPH_PATH) required |= 1 << GRAPH_COL_ARG1;
  if ((mask & required) != required) return SQLITE_CONSTRAINT;
  int argv = 0;
  for (int col = GRAPH_COL_TABLE; col < GRAPH_NCOL; col++) {
    if (!seen[col]) continue;
    info->aConstraintUsage[seen[col] - 1].argvIndex = ++argv;
    info->aConstraintUsage[seen[col] - 1].omit = 1;
  }
  info->idxNum = mask;
  info->estimatedCost = 1000;
  info->estimatedRows = 1000;
  return SQLITE_OK;
}

static int graph_push(graph_cursor *c, i64 *cap, i64 a, i64 b) {
  if (c->n == *cap) {
    i64 nc = *cap ? *cap * 2 : 64;
    i64 *na = sqlite3_realloc64(c->a, nc * sizeof(i64));
    if (!na) return SQLITE_NOMEM;
    c->a = na;
    i64 *nb = sqlite3_realloc64(c->b, nc * sizeof(i64));
    if (!nb) return SQLITE_NOMEM;
    c->b = nb;
    *cap = nc;
  }
  c->a[c->n] = a;
  c->b[c->n] = b;
  c->n++;
  return SQLITE_OK;
}

/** Level-synchronous BFS; rows are (node, depth) in visit order. */
static int graph_run_bfs(graph_cursor *c, GraphCsr *g, i64 start, i64 maxDepth) {
  i64 cap = 0;
  int rc = graph_push(c, &cap, start, 0);
  i64 s = graph_find(g, start);
  if (rc != SQLITE_OK || s < 0) return rc;
  u32 gen = graph_next_gen(g);
  g->tag[s] = gen;
  // The output rows double as the queue: each level is a contiguous range.
  u32 *q = sqlite3_malloc64(((u64)g->nNode + 1) * sizeof(u32));
  if (!q) return SQLITE_NOMEM;
  u64 head = 0, tail = 0;
  q[tail++] = (u32)s;
  for (i64 depth = 1; head < tail && (maxDepth < 0 || depth <= maxDepth); depth++) {
    u64 end = tail;
    for (; head < end; head++) {
      u32 u = q[head];
      for (u32 e = g->off[u]; e < g->off[u + 1]; e++) {
        u32 v = g->adj[e];
        if (g->tag[v] == gen) continue;
        g->tag[v] = gen;
        q[tail++] = v;
        if ((rc = graph_push(c, &cap, g->ids[v], depth)) != SQLITE_OK) goto done;
      }
    }
  }
done:
  sqlite3_free(q);
  return rc;
}

/** BFS from source until target; rows are (step, node) from source to target. */
static int graph_run_path(graph_cursor *c, GraphCsr *g, i64 source, i64 target) {
  i64 cap = 0;
  if (source == target) return graph_push(c, &cap, 0, source);
  i64 s = graph_find(g, source), t = graph_find(g, target);
  if (s < 0 || t < 0) return SQLITE_OK;
  u32 gen = graph_next_gen(g);
  u32 *q = sqlite3_malloc64(((u64)g->nNode + 1) * sizeof(u32));
  if (!q) return SQLITE_NOMEM;
  u64 head = 0, tail = 0;
  g->tag[s] = gen;
  q[tail++] = (u32)s;
  int found = 0;
  while (head < tail && !found) {
    u32 u = q[head++];
    for (u32 e = g->off[u]; e < g->off[u + 1]; e++) {
      u32 v = g->adj[e];
      if (g->tag[v] == gen) continue;
      g->tag[v] = gen;
      g->parent[v] = u;
      if (v == (u32)t) { found = 1; break; }
      q[tail++] = v;
    }
  }
  int rc = SQLITE_OK;
  if (found) {
    // Walk parents back to the source into q, then emit in forward order.
    u64 len = 0;
    for (u32 v = (u32)t;; v = g->parent[v]) {
      q[len++] = v;
      if (v == (u32)s) break;
    }
    for (u64 i = 0; i < len && rc == SQLITE_OK; i++)
      rc = graph_push(c, &cap, (i64)i, g->ids[q[len - 1 - i]]);
  }
  sqlite3_free(q);
  return rc;
}

static int graph_int_arg(graph_vtab *p, sqlite3_value *v, const char *name, i64 *out) {
  if (sqlite3_value_type(v) != SQLITE_INTEGER) {
    p->base.zErrMsg = sqlite3_mprintf("graph: %s must be an integer", name);
    return SQLITE_ERROR;
  }
  *out = sqlite3_value_int64(v);
  return SQLITE_OK;
}

static int graphFilter(sqlite3_vtab_cursor *cur, int idxNum, const char *idxStr,
                       int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxStr);
  graph_cursor *c = (graph_cursor *)cur;
  graph_vtab *p = (graph_vtab *)cur->pVtab;
  c->n = c->i = 0;
  sqlite3_value *arg[GRAPH_NCOL] = {0};
  int j = 0;
  for (int col = GRAPH_COL_TABLE; col < GRAPH_NCOL && j < argc; col++)
    if (idxNum & (1 << col)) arg[col] = argv[j++];

  const char *names[3];
  for (int col = GRAPH_COL_TABLE; col <= GRAPH_COL_DST; col++) {
    names[col - GRAPH_COL_TABLE] = (const char *)sqlite3_value_text(arg[col]);
    if (!names[col - GRAPH_COL_TABLE] || sqlite3_value_type(arg[col]) != SQLITE_TEXT) {
      p->base.zErrMsg = sqlite3_mprintf("graph: table and column names must be text");
      return SQLITE_ERROR;
    }
  }
  i64 x, y = -1;
  int rc = graph_int_arg(p, arg[GRAPH_COL_ARG0], p->kind == GRAPH_BFS ? "start" : "source", &x);
  if (rc != SQLITE_OK) return rc;
  if (arg[GRAPH_COL_ARG1] && !(p->kind == GRAPH_BFS && sqlite3_value_type(arg[GRAPH_COL_ARG1]) == SQLITE_NULL)) {
    rc = graph_int_arg(p, arg[GRAPH_COL_ARG1], p->kind == GRAPH_BFS ? "max_depth" : "target", &y);
    if (rc != SQLITE_OK) return rc;
    if (p->kind == GRAPH_BFS && y < 0) {
      p->base.zErrMsg = sqlite3_mprintf("graph: max_depth must be >= 0");
      return SQLITE_ERROR;
    }
  }

  GraphCsr *g;
  int fresh;
  char *zErr = NULL;
  rc = graph_get(p->db, p->cache, names[0], names[1], names[2], &g, &fresh, &zErr);
  if (rc != SQLITE_OK) {
    p->base.zErrMsg = zErr ? zErr : sqlite3_mprintf("graph: %s", sqlite3_errstr(rc));
    return rc == SQLITE_NOMEM ? rc : SQLITE_ERROR;
  }
  rc = p->kind == GRAPH_BFS ? graph_run_bfs(c, g, x, y) : graph_run_path(c, g, x, y);
  if (fresh) graph_csr_free(g);
  return rc;
}

static int graphNext(sqlite3_vtab_cursor *cur) {
  ((graph_cursor *)cur)->i++;
  return SQLITE_OK;
}

static int graphEof(sqlite3_vtab_cursor *cur) {
  graph_cursor *c = (graph_cursor *)cur;
  return c->i >= c->n;
}

static int graphRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  *pRowid = ((graph_cursor *)cur)->i;
  return SQLITE_OK;
}

static int graphColumn(sqlite3_vtab_cursor *cur, sqlite3_context *ctx, int i) {
  graph_cursor *c = (graph_cursor *)cur;
  if (i == GRAPH_COL_OUT0) sqlite3_result_int64(ctx, c->a[c->i]);
  else if (i == GRAPH_COL_OUT1) sqlite3_result_int64(ctx, c->b[c->i]);
  return SQLITE_OK;
}

#if SQLITE_VERSION_NUMBER >= 3044000
#define GRAPH_MODULE_TAIL 0, 0
#else
#define GRAPH_MODULE_TAIL 0
#endif
#define GRAPH_MODULE(connect)                                                  \
  {                                                                            \
    /* iVersion    */ 0, /* xCreate */ 0, connect, graphBestIndex,             \
    graphDisconnect, /* xDestroy */ 0, graphOpen, graphClose, graphFilter,     \
    graphNext, graphEof, graphColumn, graphRowid, /* xUpdate .. */ 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, GRAPH_MODULE_TAIL,                                       \
  }

static sqlite3_module graph_bfsModule = GRAPH_MODULE(graph_bfsConnect);
static sqlite3_module graph_pathModule = GRAPH_MODULE(graph_pathConnect);

typedef struct { i64 src, dst, id; } GraphWitnessEdge;
typedef struct { i64 id, edge; u64 parent; } GraphWitnessNode;

static int graph_witness_tick(sqlite3 *db, i64 *work) {
#if SQLITE_VERSION_NUMBER >= 3041000
  if (sqlite3_is_interrupted(db)) return SQLITE_INTERRUPT;
#else
  UNUSED_PARAMETER(db);
#endif
  if (*work == 0) return SQLITE_FULL;
  --*work;
  return SQLITE_OK;
}

/* One arena bounds native payload, including the returned JSON. SQLite VM,
 * allocator bookkeeping and the caller's result copy are outside this budget. */
static void graph_witness(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  sqlite3_stmt *stmt = NULL;
  unsigned char *arena = NULL;
  int rc = SQLITE_OK;
  const char *err = NULL;
  const char *names[4];
  for (int i = 0; i < 4; i++) {
    if (sqlite3_value_type(argv[i]) != SQLITE_TEXT) {
      err = "graph_witness: names must be text"; goto done;
    }
    names[i] = (const char *)sqlite3_value_text(argv[i]);
    if (!names[i]) { rc = SQLITE_NOMEM; goto done; }
    if ((int)strlen(names[i]) != sqlite3_value_bytes(argv[i])) {
      err = "graph_witness: names must not contain NUL"; goto done;
    }
  }
  for (int i = 4; i < 8; i++) {
    if (sqlite3_value_type(argv[i]) != SQLITE_INTEGER) {
      err = "graph_witness: endpoints and budgets must be integers"; goto done;
    }
  }
#if SQLITE_VERSION_NUMBER >= 3041000
  if (sqlite3_libversion_number() < 3041000) {
    err = "graph_witness requires SQLite 3.41 or later"; goto done;
  }
#else
  err = "graph_witness requires SQLite 3.41 or later"; goto done;
#endif
  i64 source = sqlite3_value_int64(argv[4]);
  i64 target = sqlite3_value_int64(argv[5]);
  i64 work = sqlite3_value_int64(argv[6]);
  i64 bytes = sqlite3_value_int64(argv[7]);
  if (work < 1 || bytes < 128 || bytes > 64 * 1024 * 1024) {
    err = "graph_witness: positive work and 128..67108864 bytes required"; goto done;
  }
  if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
  for (int i = 1; i < 4; i++) {
    if (!graph_has_column(db, names[0], names[i])) {
      err = "graph_witness: missing table or column"; goto done;
    }
  }
  arena = sqlite3_malloc64((u64)bytes);
  if (!arena) { rc = SQLITE_NOMEM; goto done; }
  if (sqlite3_msize(arena) > (u64)bytes) { rc = SQLITE_TOOBIG; goto done; }
  char *sql = sqlite3_mprintf("SELECT \"%w\",\"%w\",\"%w\" FROM \"%w\"",
                              names[1], names[2], names[3], names[0]);
  if (!sql) { rc = SQLITE_NOMEM; goto done; }
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) goto done;
  GraphWitnessEdge *edges = (GraphWitnessEdge *)arena;
  u64 ne = 0;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
    for (int i = 0; i < 3; i++) {
      if (sqlite3_column_type(stmt, i) != SQLITE_INTEGER) {
        err = "graph_witness: endpoints and edge IDs must be non-NULL integers";
        goto done;
      }
    }
    i64 id = sqlite3_column_int64(stmt, 2);
    for (u64 i = 0; i < ne; i++) {
      if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
      if (edges[i].id == id) { err = "graph_witness: duplicate edge ID"; goto done; }
    }
    if ((ne + 1) * sizeof(*edges) > (u64)bytes) { rc = SQLITE_TOOBIG; goto done; }
    edges[ne++] = (GraphWitnessEdge){sqlite3_column_int64(stmt, 0),
                                   sqlite3_column_int64(stmt, 1), id};
  }
  if (rc != SQLITE_DONE) goto done;
  sqlite3_finalize(stmt); stmt = NULL;
  u64 used = ne * sizeof(*edges);
  GraphWitnessNode *nodes = (GraphWitnessNode *)(arena + used);
  if (used + sizeof(*nodes) > (u64)bytes) { rc = SQLITE_TOOBIG; goto done; }
  nodes[0] = (GraphWitnessNode){source, 0, 0};
  u64 nn = 1, head = 0, found = source == target ? 0 : (u64)-1;
  while (head < nn && found == (u64)-1) {
    for (u64 e = 0; e < ne; e++) {
      if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
      if (edges[e].src != nodes[head].id) continue;
      u64 n;
      for (n = 0; n < nn; n++) {
        if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
        if (nodes[n].id == edges[e].dst) break;
      }
      if (n != nn) continue;
      if (used + (nn + 1) * sizeof(*nodes) > (u64)bytes) { rc = SQLITE_TOOBIG; goto done; }
      nodes[nn] = (GraphWitnessNode){edges[e].dst, edges[e].id, head};
      if (edges[e].dst == target) found = nn;
      nn++;
      if (found != (u64)-1) break;
    }
    head++;
  }
  used += nn * sizeof(*nodes);
  u64 len = 0;
  if (found != (u64)-1) {
    for (u64 n = found;; n = nodes[n].parent) {
      if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
      len++;
      if (n == 0) break;
    }
  }
  /* Signed int64 JSON pairs need at most 46 bytes per node. */
  if (used + len * 48 + 3 > (u64)bytes) { rc = SQLITE_TOOBIG; goto done; }
  char *out = (char *)arena + used;
  char *p = out;
  *p++ = '[';
  for (u64 step = 0; step < len; step++) {
    u64 n = found;
    for (u64 j = step + 1; j < len; j++) {
      if ((rc = graph_witness_tick(db, &work)) != SQLITE_OK) goto done;
      n = nodes[n].parent;
    }
    sqlite3_snprintf(48, p, step ? ",[%lld,%lld]" : "[%lld,null]",
                     nodes[n].id, nodes[n].edge);
    p += strlen(p);
  }
  *p++ = ']'; *p = 0;
  sqlite3_result_text(ctx, out, (int)(p - out), SQLITE_TRANSIENT);
  rc = SQLITE_OK;
done:
  sqlite3_finalize(stmt);
  sqlite3_free(arena);
  if (err || rc != SQLITE_OK) {
    sqlite3_result_error(ctx, err ? err : rc == SQLITE_FULL ? "graph_witness: work limit exceeded" :
                         rc == SQLITE_TOOBIG ? "graph_witness: memory limit exceeded" : sqlite3_errstr(rc), -1);
    if (!err) sqlite3_result_error_code(ctx, rc);
  }
}

/** Registers graph_bfs and graph_shortest_path with one shared per-connection cache. */
static int graph_register(sqlite3 *db, char **pzErrMsg) {
  if (sqlite3_libversion_number() < 3037000) {
    *pzErrMsg = sqlite3_mprintf("graph_bfs requires SQLite 3.37.0 or later");
    return SQLITE_ERROR;
  }
  GraphCache *c = sqlite3_malloc(sizeof(*c));
  if (!c) return SQLITE_NOMEM;
  memset(c, 0, sizeof(*c));
  // The cache is freed with the last module that references it.
  int rc = sqlite3_create_module_v2(db, "graph_bfs", &graph_bfsModule, c, NULL);
  if (rc == SQLITE_OK)
    rc = sqlite3_create_module_v2(db, "graph_shortest_path", &graph_pathModule, c,
                                  graph_cache_free);
  else
    sqlite3_free(c);
  if (rc == SQLITE_OK)
    rc = sqlite3_create_function_v2(db, "graph_witness", 8, SQLITE_UTF8 | SQLITE_DIRECTONLY,
                                    NULL, graph_witness, NULL, NULL, NULL);
  if (rc != SQLITE_OK)
    *pzErrMsg = sqlite3_mprintf("Error creating graph modules: %s", sqlite3_errmsg(db));
  return rc;
}

#endif /* SQLITE_VEC_GRAPH_C */
