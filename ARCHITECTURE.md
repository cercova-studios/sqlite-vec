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

Connections cache these rows in memory (`cache_mb`). The cache is cleared on
`xRollback` and `xRollbackTo`, and when the pager's data version shows a commit
by another connection. vec0 sets `xSavepoint` only so that SQLite calls
`xRollbackTo` when a statement fails inside a transaction.

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
