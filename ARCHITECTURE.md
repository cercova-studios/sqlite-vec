# `sqlite-vec` Architecture

Internal documentation for how `sqlite-vec` works under-the-hood. Not meant for
users of the `sqlite-vec` project, consult
[the official `sqlite-vec` documentation](https://alexgarcia.xyz/sqlite-vec) for
how-to-guides. Rather, this is for people interested in how `sqlite-vec` works
and some guidelines to any future contributors.

Very much a WIP.

## Builds

- **C core** (`sqlite-vec.c`, `sqlite-vec-*.c`): `make loadable`, `static`,
  `amalgamation`. It depends only on SQLite, so it can ship as a single file.
- **Hybrid** (`lance/`, `make hybrid` → `dist/vec0_lance`): a Rust crate whose
  `build.rs` compiles the C core in, so one library registers `vec0` and
  `lance0`. Dependencies point only one way: Rust may use the C core, but the
  C core never depends on Rust, Lance or protoc.

`make test-unit` builds and runs the C unit tests with the bundled SQLite
amalgamation and experimental IVF enabled. `tests/test-unit.c` includes the C
core directly, so tests use the production internal types and definitions.
The harness rejects `NDEBUG` builds to prevent silently disabling assertions.

`lance0` follows the same transaction rules as the vec0 index caches: rollback
and `xRollbackTo` undo pending state, and SQLite's shadow table
`<name>_lance` records which Lance version is committed (see `lance/README.md`).
`tests/test-hybrid.py` checks the shared vector format, queries that join both
table types, and transactions that write to both.

### Storage providers

`sqlite3_vec_storage_methods` (in `sqlite-vec.h`) lets a vec0 table keep its
auxiliary columns outside the `<table>_auxiliary` shadow table, declared with
`storage=<name>` and an optional `storage_uri='...'`. A provider registers per
connection with `sqlite3_set_clientdata(db, "sqlite-vec-storage:<name>", ...)`
(SQLite 3.44+), so the C core has no global state and no link-time dependency
on any provider. Without a registered provider, CREATE fails with a message
naming the build that has one.

vec0 forwards all writes, reads (`xColumn`, plus `xPrefetch` with a KNN
query's result rowids) and transaction callbacks (`xSync`, `xCommit`,
`xRollback`, `xSavepoint`, `xRelease`, `xRollbackTo`) to the provider. vec0 sets
`xSavepoint`/`xRelease` so SQLite reports statement rollbacks to it.

The Lance provider (`lance/src/provider.rs`) and `lance0` share
`lance/src/store.rs` (`LanceStore`), the one implementation of the Lance commit
rules. Its shadow table `<table>_storage` records the committed Lance version and
the dataset location.

Storage ABI version 2 adds the optional `xCommand` callback. The C core checks
`iVersion >= 2` before reading it; providers using version 1 still support their
original operations. Commands unknown to the C indexes are forwarded to storage.
The Lance provider and `lance0` accept `optimize`, which requests file compaction
and rowid-index maintenance during `xSync`. Savepoint rollback restores the
request state, and transaction rollback discards it. Maintenance retains old
Lance versions so readers and failed commits can still use the recorded version.

For `lance0`, `optimize` also builds an IVF_HNSW_SQ index named `vector_idx` on
`vector` at 16,384 rows, or updates that index if it already exists. This is
opt-in: writes alone never create it. `LanceStore::ann_column` is set only by
`lance0`; vec0 storage providers keep their existing C index behavior. Vector
maintenance follows compaction in `xSync` and shares its recorded-version rules.
Lance scans unindexed changes alongside indexed data; the virtual table merges
pending writes as before.

The index uses L2, 8-bit scalar quantization, Lance's default HNSW construction
parameters, and `clamp(rows / 256, 1, 256)` IVF partitions. Search starts at 32
partitions, permits adaptive expansion for prefilters, and reranks four times
the requested candidates using original vectors (`ef >= max(128, 4*k)`). Returned
distances remain L2; indexed neighbor selection is approximate. Small tables and
tables without a vector index continue to scan exactly.

## Optional search adapter

`examples/search-lab/src/lib.rs` exposes one host-side `Adapter::call` boundary
through JSON/JSONL CLI records and local stdio MCP tools in `src/main.rs`.
The optional Rust binary uses `rusqlite` with bundled SQLite and the official
`rmcp` SDK. Its dependencies are isolated from the C and hybrid builds. Python
is used only for fixture generation and independent tests. Trusted host configuration
selects the database, native extension, corpus tables and fixed tenant/time scope.
The adapter loads the extension once per process, disables further loading and
enforces read-only access with SQLite read-only open flags, `query_only` and an
authorizer. A scoped extension-loading guard disables further loading even when
loading fails. One serialized connection serves both transports.

Each operation uses one explicit read transaction. Separate calls may observe
different commits. Search ranks eligible flat float/L2 vectors with scalar distance
SQL for deterministic cutoff ties, combines branch ranks with RRF, and fetches
evidence by stable corpus/document IDs. Traversal reuses distinct-node native BFS
over a configured edge source after checking its size and endpoint eligibility in
the same snapshot. Admission limits restrict supported workloads; they do not
provide hard native memory, work or cancellation budgets. See the adapter
[contract and runnable example](examples/search-lab/README.md).

## Graph traversal

`sqlite-vec-graph.c` registers `graph_bfs` and `graph_shortest_path` over a selected
ordinary edge table/view. Native node IDs are signed 64-bit integers scoped by the
selected source and columns; no catalog or type binder is implied. BFS emits each
reachable node once with minimum hop distance and includes its anchor at zero,
even when absent from the edge source. The path helper returns one shortest node
sequence, including a zero-step sequence for equal endpoints. Neither helper
returns edge identities or path multiplicities.

Each connection caches up to eight CSR graphs. Write transactions rebuild rather
than reuse cached adjacency so own writes and savepoint rollback remain visible.
Native traversal materializes its output before SQL limits apply. The search-lab
[graph contract and fixtures](examples/search-lab/graph-semantics.md) distinguish
these native semantics from catalog admission, typed fixture identities, scoped
subgraphs and SQL bag/optional-binding behavior. Its independent all-pairs oracle
is test-only; it adds no runtime dependency or GraphIR execution surface.

## `vec0`

### Shadow Tables

#### `xyz_chunks`

- `chunk_id INTEGER`
- `size INTEGER`
- `validity BLOB`
- `rowids BLOB`

#### `xyz_rowids`

- `rowid INTEGER`
- `id`
- `chunk_id INTEGER`
- `chunk_offset INTEGER`

#### `xyz_vector_chunksNN`

- `rowid INTEGER`
- `vector BLOB`

#### `xyz_auxiliary`

- `rowid INTEGER`
- `valueNN [type]`

#### `xyz_metadatachunksNN`

- `rowid INTEGER`
- `data BLOB`

#### `xyz_metadatatextNN`

- `rowid INTEGER`
- `data TEXT`

### Numeric metadata zone maps

For tables with the modern hidden command column, the existing
`xyz_info(key TEXT PRIMARY KEY, value ANY)` table also holds
`zonemap_<metadata-index:02d>_<chunk-id>` entries for INTEGER, FLOAT and BOOLEAN
metadata in new chunks. Each populated entry is a 16-byte native-format BLOB:
minimum and maximum as two int64 values, or two doubles for FLOAT. A NULL entry
marks a new empty chunk until its first metadata write. TEXT keeps its normal
filtering path; partition predicates already filter `xyz_chunks` in SQL.

`vec0_new_chunk` seeds these entries, `vec0_write_metadata_value` widens them,
and whole-chunk deletion removes them. Ordinary deletes leave bounds unchanged:
the validity bitmap removes deleted rows, so wider bounds only cost pruning
opportunities. The stats writes share SQLite's statement, savepoint and transaction
rollback. No stats values are cached in memory; the two cached prepared statements
are finalized by `vec0_free_resources`, including on rename.

`vec0Filter_knn_chunks_iter` rejects a chunk before opening metadata blobs only
when a bound proves a numeric comparison or integer IN filter cannot match.
Integer comparisons retain all 64 bits. Missing, NULL, invalid-length or reversed
bounds use normal filtering. Existing chunks without stats are not backfilled,
and writes never initialize missing stats from a single row of an existing chunk.
This pruning applies to the flat chunk scan; index-specific ANN paths are unchanged.

Before writing a database with an older extension that does not maintain zone
maps, remove its `zonemap_*` entries from each vec0 `_info` table. Otherwise a
later reader could trust bounds made stale by those writes. Direct shadow-table
metadata writes likewise bypass maintenance and are unsupported.

#### `xyz_hnsw_nodesNN`

One row per vector in an `indexed by hnsw(...)` column.

- `rowid INTEGER PRIMARY KEY`
- `level INTEGER`: the node's top HNSW layer
- `vector BLOB`
- `UNIQUE (level, rowid)`: the entry point is the first row of this index in
  descending order, so it is never stored separately

#### `xyz_hnsw_edgesNN`

`WITHOUT ROWID`, one row per node per layer.

- `rowid INTEGER`, `level INTEGER` (primary key)
- `neighbors BLOB`: packed `i64` rowids

Connections cache these rows in memory (`cache_mb`). Node/vector writes remain
immediate. Neighbor-list edits update the cache and mark the list dirty, so
queries and subsequent graph edits see uncommitted changes without replaying a
separate operation queue. Each dirty list is written once per flush, during
`xSync`, `xSavepoint`, before eviction or a cache-setting change,
and when a rebuild finishes. `cache_mb=0` retains immediate neighbor writes.
The cache budget is checked between operations, so one operation can exceed it.

`xSavepoint` flushes older dirty lists before the new SQLite rollback boundary;
statement savepoints therefore bound batching too. Full transaction rollback
and ordinary `ROLLBACK TO` clear the cache after SQLite restores shadow tables,
including flushed lists. A nested edge-write rollback during an active flush
retains every HNSW column's cache, so retry preserves all pending lists.
Full rollback always clears caches, including during a flush; cache objects
remain allocated while their contents are reset. Flush errors propagate to
the caller; a failed `xSync` aborts the transaction. Dirty flags are cleared
only after the entire cache flush succeeds. The pager's data version
invalidates cached rows after another connection commits. Deleted nodes have
their pending lists canceled, so a
later flush cannot recreate their edges. This changes write scheduling, not
the persisted graph format or neighbor-selection algorithm.

`xRelease` leaves dirty lists pending: SQLite has already released its
shadow-table savepoint before this callback, so it is too late for a fallible
flush there. Opening a new savepoint flushes prior edits before that boundary.

### DiskANN search visits

`diskann_search` bounds its candidate list by the search-list size, but can
visit more unique nodes than that size. Its per-search `DiskannVisitedSet`
grows when half full and retains every visited rowid, including zero. Search
records a visit before reading the node's edges, so dropped candidates and
missing nodes cannot cause repeated expansion.

Visited-set memory grows with unique visits. Resizing temporarily holds both
slot arrays; allocation failure returns `SQLITE_NOMEM` and frees the search
state. Failed growth leaves the existing set intact. This state is local to
one search and does not change the persisted graph format or SQL options.

### Flat-vector write batching

For vector columns stored in `xyz_vector_chunksNN`, `Vec0FlatWrite` buffers
one contiguous byte range per column, up to 64 KiB. Consecutive writes to the
same chunk append to that range. A chunk switch, gap, overlap or full buffer
flushes the prior range through one blob write. Chunk/rowid validity, rowid
mappings, metadata and auxiliary writes remain immediate SQLite writes.
Blob lengths are checked when a range starts and again when it flushes.

Reads (`xFilter` and `xColumn`), commands, `xSavepoint`, `xSync`, rename and
deletion flush pending vector bytes before using the existing read/write
paths. This preserves read-your-writes; a read can incur flush latency or
report a write error. `xRelease` leaves pending ranges alone. SQLite rolls
back flushed bytes, while `xRollback` and `xRollbackTo` discard later buffered
bytes. An active flush owns its detached range across SQLite callbacks, and
a transaction generation prevents restoring data after a full rollback.

The existing limit of 16 vector columns bounds payload memory to 1 MiB per
table; smaller chunks allocate smaller buffers. Buffers are cleared after
successful flushes and do not cache other connections' committed data.
Storage providers, rescore, IVF and graph indexes keep their own write paths.
The persisted vector/chunk format and public SQL options are unchanged.

### IVF cell write batching

The experimental IVF build stores up to 64 vectors per packed cell in
`xyz_ivf_cellsNN`, with validity bits, rowids and quantized or original vectors.
`IvfWriteBuffer` holds at most 64 changed cells and 8 MiB per table across vector
columns. Inserts choose an unused validity bit; deletions clear that bit and
decrement the live count. The count is not a slot offset. Rowid mappings and
quantized columns' original-vector KV writes remain immediate SQLite writes.

`ivf_flush` writes each buffered cell with one UPDATE and clears the buffer
after success. It runs in `xSync`, before opening savepoints, before reads,
before rename, around centroid commands and when a new cell would exceed the
buffer limit. Reads therefore see pending writes through the existing query
paths. A read can incur write latency or report a flush error. No cell values
remain cached after a successful flush, so other connections' commits require
no separate cell-cache invalidation protocol.

`xRollback` and `xRollbackTo` discard pending cells after SQLite restores shadow
tables. `xRelease` does not flush. A flush owns its detached buffer while nested
SQLite statements run: a nested statement rollback preserves the pending
logical writes for retry, while a transaction rollback advances the buffer
generation and discards them. Flush failures cannot leave freed cells in use
or silently discard earlier pending vectors. Persisted cell formats, centroid
selection and quantizers are unchanged.

### idxStr

The `vec0` idxStr is a string composed of single "header" character and 0 or
more "blocks" of 4 characters each.

The "header" charcter denotes the type of query plan, as determined by the
`enum vec0_query_plan` values. The current possible values are:

| Name                       | Value | Description                                                            |
| -------------------------- | ----- | ---------------------------------------------------------------------- |
| `VEC0_QUERY_PLAN_FULLSCAN` | `'1'` | Perform a full-scan on all rows                                        |
| `VEC0_QUERY_PLAN_POINT`    | `'2'` | Perform a single-lookup point query for the provided rowid             |
| `VEC0_QUERY_PLAN_KNN`      | `'3'` | Perform a KNN-style query on the provided query vector and parameters. |

Each 4-character "block" is associated with a corresponding value in `argv[]`.
For example, the 1st block at byte offset `1-4` (inclusive) is the 1st block and
is associated with `argv[1]`. The 2nd block at byte offset `5-8` (inclusive) is
associated with `argv[2]` and so on. Each block describes what kind of value or
filter the given `argv[i]` value is.

A KNN plan needs its arguments: the MATCH on a vector column, `k`, and any
hidden parameter column declared after `k`. When one of them reads another table
of the query, SQLite also weighs plans that scan `vec0` before that table, in
which the argument is unusable. `xBestIndex` returns `SQLITE_CONSTRAINT` for
such a plan (`vec0_constraint_is_knn_argument`), so the planner takes one that
gives `vec0` every argument.

Filters on rowid, partition key, metadata and distance columns are left to the
plan. A filter whose value comes from another table is unusable in a plan that
scans `vec0` first, and SQLite takes such a plan whenever the query allows it,
in either FROM order, applying the filter to the k rows `vec0` returns. That
keeps a lookup join at k rows, and `vec0` cannot tell a lookup join from a
filter join: in

```sql
select v.rowid, c.name from v join categories c on c.id = v.category
where v.embedding match ? and k = 10
```

refusing the plan without `v.category = c.id` would run one KNN per
`categories` row and return 10 rows per category. A query that reads the other
table first, because that table supplies a KNN argument, a `CROSS JOIN` names it
first, or `vec0` is on the right of a `LEFT JOIN` with the filter in `ON`, has
the filter usable in every plan, so `vec0` runs once per row of that table and
applies the filter before the k. So does a correlated subquery, whose outer
columns are constants to `vec0`.

#### `VEC0_IDXSTR_KIND_KNN_MATCH` (`'{'`)

`argv[i]` is the query vector of the KNN query.

The remaining 3 characters of the block are `_` fillers.

#### `VEC0_IDXSTR_KIND_KNN_K` (`'}'`)

`argv[i]` is the limit/k value of the KNN query: the `k` of a `k = n`
constraint, or the `LIMIT n` of an `ORDER BY distance LIMIT n` query
(`SQLITE_INDEX_CONSTRAINT_LIMIT`, SQLite 3.38+; the planner only offers it
to a solitary `vec0` scan, so in joins use `k = n`). Exactly one of the two
must be present: both given, or neither, is an error at plan time — a KNN
query is never silently unbounded. A negative value (`LIMIT -1`) and a
`LIMIT + OFFSET` sum over the 4096 cap are errors at filter time.

The remaining 3 characters of the block are `_` fillers.

#### `VEC0_IDXSTR_KIND_KNN_OFFSET` (`'('`)

`argv[i]` is the `OFFSET m` of a `LIMIT n OFFSET m` KNN query, present only
after a `VEC0_IDXSTR_KIND_KNN_K` block sourced from `LIMIT`. `vec0` fetches
`n + m` rows; the constraint is passed with `omit = 0` so SQLite itself
skips the first `m` rows of the result.

The remaining 3 characters of the block are `_` fillers.

#### `VEC0_IDXSTR_KIND_KNN_ROWID_IN` (`'['`)

`argv[i]` is the optional `rowid in (...)` value, and must be handled with
[`sqlite3_vtab_in_first()` / `sqlite3_vtab_in_next()`](https://www.sqlite.org/c3ref/vtab_in_first.html).

The remaining 3 characters of the block are `_` fillers.

#### `VEC0_IDXSTR_KIND_KNN_PARTITON_CONSTRAINT` (`']'`)

`argv[i]` is a "constraint" on a specific partition key.

The second character of the block denotes which partition key to filter on,
using `A` to denote the first partition key column, `B` for the second, etc. It
is encoded with `'A' + partition_idx` and can be decoded with `c - 'A'`.

The third character of the block denotes which operator is used in the
constraint. It will be one of the values of `enum vec0_partition_operator`, as
only a subset of operations are supported on partition keys.

The fourth character of the block is a `_` filler.

#### `VEC0_IDXSTR_KIND_POINT_ID` (`'!'`)

`argv[i]` is the value of the rowid or id to match against for the point query.

The remaining 3 characters of the block are `_` fillers.

#### `VEC0_IDXSTR_KIND_METADATA_CONSTRAINT` (`'&'`)

`argv[i]` is the value of the `WHERE` constraint for a metdata column in a KNN
query.

The second character of the block denotes which metadata column the constraint
belongs to, using `A` to denote the first metadata column column, `B` for the
second, etc. It is encoded with `'A' + metadata_idx` and can be decoded with
`c - 'A'`.

The third character of the block is the constraint operator. It will be one of
`enum vec0_metadata_operator`, as only a subset of operators are supported on
metadata column KNN filters.

The foruth character of the block is a `_` filler.
