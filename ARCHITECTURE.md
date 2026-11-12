# `sqlite-vec` Architecture

Internal documentation for how `sqlite-vec` works under-the-hood. Not meant for
users of the `sqlite-vec` project, consult
[the official `sqlite-vec` documentation](https://alexgarcia.xyz/sqlite-vec) for
how-to-guides. Rather, this is for people interested in how `sqlite-vec` works
and some guidelines to any future contributors.

Very much a WIP.

## Builds

`flake.nix` and its committed `flake.lock` define an optional native contributor
shell for Linux and macOS on x86_64 and aarch64. Nix supplies the compiler, Make,
Git, Python, gettext, curl and unzip; existing Make targets and bootstrap checks
remain the build contract. Entering the shell does not fetch SQLite vendor
sources or build the project. The lock pins tool inputs; host SIMD selection,
explicit vendor acquisition and working-tree sources remain separate inputs.
Rust/Lance, model runtimes and Python test packages are outside this shell.

`scripts/bootstrap.py` provides a read-only prerequisite report by default.
Its `dev --output PATH` route builds working-tree C sources through existing Make
targets in a fresh output directory, then runs C unit and in-memory vector/FTS5
checks. It supplies FTS5 flags through the environment so Make retains host SIMD
settings, retains a full log and never installs tools or downloads
dependencies. Make may generate the ignored root header. Existing destinations
are rejected; failed attempts retain their outputs for diagnosis. This contributor
route is separate from committed-source local bundles described below. See
[portable agent onboarding](docs/agent-onboarding.md).

The explicit `scripts/vendor.sh` setup downloads SQLite 3.45.3 into a temporary
directory, fails on download/HTTP/extraction errors, and copies extracted files
into `vendor/` only after extraction succeeds. It resolves the checkout from its
own path and removes its temporary download on exit.

- **C core** (`sqlite-vec.c`, `sqlite-vec-*.c`): `make loadable`, `static`,
  `amalgamation`. It depends only on SQLite, so it can ship as a single file.
- **Hybrid** (`lance/`, `make hybrid` → `dist/vec0_lance`): a Rust crate whose
  `build.rs` compiles the C core in, so one library registers `vec0` and
  `lance0`. Dependencies point only one way: Rust may use the C core, but the
  C core never depends on Rust, Lance or protoc.

The hybrid build also compiles `lance/src/vtab.c`, a private bridge for the
variadic `sqlite3_vtab_config` call. The loadable Rust FFI wrapper omits its
variadic argument; the bridge enables SQLite constraint handling for `lance0`.

`make test-unit` builds and runs the C unit tests with the bundled SQLite
amalgamation and experimental IVF enabled. `tests/test-unit.c` includes the C
core directly, so tests use the production internal types and definitions.
The harness rejects `NDEBUG` builds to prevent silently disabling assertions.

Native loadable/static/amalgamation builds track every included `sqlite-vec-*.c`
feature file. `make static` stages `dist/sqlite-vec.h` beside the archive. Build
directories are order-only prerequisites, so directory timestamps do not force
recompilation. The Rust binding commits a standalone Cargo manifest, C
amalgamation, extension header, SQLite 3.45.3 headers and licenses. Clean-clone
and revision-pinned Git dependency builds require Cargo and a C compiler only.
Maintainers refresh those inputs with `make -C bindings/rust deps` after
regenerating the root header; consumers do not run Make. Cargo watches the
committed C/header inputs. `tests/rust-packaging.py` checks freshness against
the modular sources, bundled-rusqlite registration, Cargo archive completeness,
a clean pinned Git consumer and flat vec0 file interchange with registry 0.1.9.
`make test-packaging` exercises isolated static and Rust-staged C consumers and
included-feature invalidation, without modifying the checkout's generated files.
`make cli` builds `$(prefix)/sqlite3` through one file target using the existing
direct C recipe. It tracks included feature sources, vendor C/headers and the
connection initializer; an unchanged CLI is not compiled again. The output
directory is order-only. `make test-unit` still compiles and executes its assertions
whenever requested, including by the contributor bootstrap.
As with the other native file targets, changed compiler flags/profiles require a
fresh output directory; Make does not compare old and new command arguments.

The Rust binding exposes the C initializer with its three pointer arguments and
SQLite result code. Callers can register on each live connection from their linked
SQLite runtime. The static archive contains no SQLite implementation; the
application supplies one runtime, including through rusqlite's bundled feature.
Apple system SQLite rejects process-global `sqlite3_auto_extension`;
per-connection registration works without dynamic extension loading. Tests retain
auto-extension registration on hosts that support it. See
[Rust registration](bindings/rust/README.md).

`scripts/local_bundle.py` builds a project-local bundle from committed HEAD in a
temporary source tree and fresh Cargo target directory. It stages the native
SQLite CLI (vendor SQLite with FTS5), the existing read-only Rust CLI/MCP adapter,
loadable/static libraries, C amalgamations/headers, licenses and usage documents.
The default and experimental IVF profiles use isolated builds. Core dependencies
remain SQLite-only; the adapter continues to bundle its own SQLite version.
The manifest records both SQLite versions, native flags/capabilities, compiler
versions, exact source/vendor input hashes and installed artifact hashes/sizes.
The builder, guide and `examples/bundle-consumer/app.py` starter are copied from
the invoking checkout; their artifact hashes identify those bytes separately from
the compiled source revision. The installed `examples/app.py` runs from beside
the bundle, using its native CLI against fresh project-local fixtures. Independent
Python distances check KNN results; reopen, rollback and backup/restore are checked
without a model, Python SQLite extension loading or source-checkout dependency.
Existing destinations are rejected; a manifest is written last as the completion
marker. Failed builds retain their log. Installed verification reads hashes only;
it does not authenticate an untrusted bundle. See [local bundles](docs/local-bundle.md)
and `tests/bundle.py` for relocation, failure and backup/restore checks.

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

Each store retains the SQLite database name supplied by its virtual-table callback.
Metadata creation, URI/version lookup, commit updates, rename and DROP qualify the
database and shadow-table identifiers separately. Same-named tables in `main`,
attached databases and `temp` therefore keep separate version metadata. Attachment
aliases are connection-local; reopening under a different alias uses that schema's
metadata. Transaction and savepoint callbacks keep their existing pending/undo rules.

`lance0` checks rowid uniqueness against pending changes and the recorded Lance
version before changing rows. Its ID-only lookup does not read vector/BLOB payloads.
SQLite handles ABORT, IGNORE, FAIL and ROLLBACK conflicts; REPLACE overwrites the
existing identity. Rowid-changing updates preserve unchanged column values. vec0
already enforces identity through its SQLite rowid map and needs no extra lookup.
Rowid filter literals use an explicit integer cast for `i64::MIN`, which Lance's
parser otherwise rounds through floating point.

The provider checks SQLite conversion buffers before constructing Rust slices,
preserves `SQLITE_NOMEM`, and rejects invalid UTF-8 TEXT. SQL NULL, empty TEXT and
empty BLOB remain distinct. Conversion failure stages no provider mutation.
When SQLite first enlists a store inside nested savepoints, the store records the
unannounced outer savepoints too. Releasing the inner scope cannot discard the undo
history needed by an outer rollback.

Both entrypoints route DROP through `LanceStore::destroy`: only the SQLite version
table is removed, so rollback and WAL readers retain their dataset files. Committed
DROP also retains local/remote files; no automatic filesystem cleanup is performed.
DROP refuses uncommitted writes/maintenance to that store because a recreated vtab
cannot recover its pending map. vec0 calls the provider before changing other shadow
tables so that refusal cannot invalidate pending state. A failed `vec0Destroy`
retains the live vtab; only successful destruction frees it. `lance0` removes its
`<table>_lance` version table through the same lifecycle method.

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

## Experiment archive

`benchmarks-ann/experiments/README.md` indexes written reproduction instructions
and historical results. Completed experiment directories contain Markdown only;
scripts, lockfiles, raw measurements and generated artifacts are recovered from
the pinned pre-cleanup Git snapshot when needed. Current tests, application
examples and reusable benchmark tools remain in their normal source locations.
Active experiments keep their required scratch inputs isolated until completion;
they are excluded from completed-experiment teardown.

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
loading fails. One serialized connection serves both transports. MCP executes handlers on a blocking worker so the protocol loop can receive cancellation.

Each operation uses one explicit read transaction. Separate calls may observe
different commits. Search ranks eligible flat float/L2 vectors with scalar distance
SQL for deterministic cutoff ties. Keyword eligibility uses correlated `EXISTS`
inside the FTS branch before its candidate limit; vector eligibility retains
`IN`. The adapter combines branch ranks with RRF and fetches
evidence by stable corpus/document IDs. Traversal reuses distinct-node native BFS
over a configured edge source after checking its size and endpoint eligibility in
the same snapshot. Admission limits restrict supported workloads; they do not
provide hard native memory or work budgets for BFS. The typed `witness` operation uses the bounded native evidence helper with per-call budgets and MCP cancellation. See the adapter
[contract and runnable example](examples/search-lab/README.md).

## Experimental task writer

`examples/write-prototype` is a separate Rust library over fresh ordinary SQLite
task fixtures. `Writer` owns one write connection; it does not share or weaken the
read-only adapter connection. `BEGIN IMMEDIATE` serializes revision checks, task
mutations, monotonic sidecar versions/tombstones, Session capture, changed-row
indexes and immutable retry receipts. Undo inverts canonical ordinary values only and
allocates a new revision. Uncertain commit/rollback outcomes discard the connection.
Its schema marker/version checks reject unsupported fixtures; they are not an audit
security boundary. Byte/row limits bound admitted data and capture output, not SQLite
RSS. The library has no transport routes.

Fresh schema-version-2 indexed fixtures opt into host-selected model identity and
dimensions through unsafe constructors that load a trusted ABI-compatible vec0
library. Prepared embeddings are computed outside the transaction. Canonical
`source_vectors` records little-endian float32 values/model IDs beside task text;
Session captures these two ordinary tables only. FTS5 `lexical`, exact L2 vec0
`vectors` and `index_state` are rebuilt per source change in the same transaction,
including selective undo. Logical derived rows are validated before managed writes
and source reads. Source epochs advance on content/vector/liveness changes, while
claim/undo-claim changes only task revision and never rewrites indexes. Derived
state has no foreign key into transient Session application; canonical foreign keys
remain checked. There is no existing-data migration or model runner.

The prototype pins rusqlite 0.40.2 because the adapter's 0.41 Session feature currently
cannot resolve bindgen 0.73. The adapter's dependencies and default tools remain
unchanged. See [prototype scope and checks](examples/write-prototype/README.md) for
the version-split removal condition, failure behavior and resource limits.

## Graph traversal

`sqlite-vec-graph.c` registers `graph_bfs` and `graph_shortest_path` over a selected
ordinary edge table/view. Native node IDs are signed 64-bit integers scoped by the
selected source and columns; no catalog or type binder is implied. BFS emits each
reachable node once with minimum hop distance and includes its anchor at zero,
even when absent from the edge source. The path helper returns one shortest node
sequence, including a zero-step sequence for equal endpoints. Neither helper
returns edge identities or path multiplicities.

Each connection caches up to eight CSR graphs with at most 16 MiB of resident adjacency allocations, measured with `sqlite3_msize`. LRU eviction enforces both bounds; an oversized graph is used uncached. Build scratch, the new graph before admission and cursor output are outside the resident-cache bound. `graph_cache_bytes()` reports retained adjacency bytes. Write transactions rebuild rather
than reuse cached adjacency so own writes and savepoint rollback remain visible.
Native traversal materializes its output before SQL limits apply. The search-lab
[graph contract and fixtures](examples/search-lab/graph-semantics.md) distinguish
these native semantics from catalog admission, typed fixture identities, scoped
subgraphs and SQL bag/optional-binding behavior. Its independent all-pairs oracle
is test-only. Optional corpus graph mappings enable the `g1/1` typed `reach`
operation, which binds one node and edge type to the configured tables, checks
their schema in the operation's read transaction, and lowers to scoped native BFS.
It supports outgoing minimum-hop reachability only; it is not a general GraphIR
compiler and adds no dependency to the C core.

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

### Flat-vector KNN reads

The flat chunk scan applies validity, rowid and metadata filters before reading
vector payloads. Empty candidate chunks are skipped. `vec0_read_vector_chunk`
uses the resulting bitmap to read contiguous runs of surviving vectors through
SQLite's incremental BLOB API, keeping each vector at its normal buffer offset.
Distance calculation, distance predicates and top-k merging use the same bitmap
and ranking code as whole-chunk reads. Partition predicates still select chunks
in SQL, and text IDs still resolve through the shared rowid-IN collection.

The reader keeps whole-chunk reads when more than one-eighth of slots survive,
or when surviving bytes plus 64 KiB per read run reach the full chunk size.
This private cost heuristic avoids many small BLOB calls; it is not a SQLite
page-size assumption or a runtime option. The existing full-chunk scratch buffer
is retained. No new cache, rowid mapping or persisted state is introduced.
`xFilter` flushes pending flat writes before either read strategy, so own writes,
savepoints and rollback continue through the existing SQLite transaction path.
BLOB length checks and read failures retain the whole-chunk error path.

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

## Bounded native graph evidence

`graph_witness(table, src_column, dst_column, id_column, source, target,
work, bytes)` returns a JSON array of `[node, incoming_edge_id]` pairs for one
unweighted shortest directed path. The first incoming edge is JSON null;
unreachable targets return `[]`. Equal endpoints return one node after validating
the source. Edge IDs must be unique non-NULL signed integers in the selected
relation; endpoints must also be non-NULL integers. Ties follow source scan order.
IDs are supplied by the host, which owns their persistence and graph scope.

The operation requires SQLite 3.41+, checks `sqlite3_interrupt` during native
loops, and consumes one work unit per scanned row or native comparison/visit,
including hash probes, index initialization and path output.
Budget exhaustion fails the entire result (`SQLITE_FULL` for work,
`SQLITE_TOOBIG` for memory); interruption returns `SQLITE_INTERRUPT`.
A single arena holds edge rows, hash/list indexes, BFS queue/parents and JSON construction, bounded
by `bytes` (128 through 64 MiB). SQLite VM/schema/SQL preparation memory, allocator
bookkeeping and the caller's returned-text copy are outside this payload limit.
This is not a process RSS limit. Source SQL work requires the host's SQLite
progress handler if a separate VM budget is needed.

This direct-only scalar function rebuilds from the current statement snapshot
and keeps no adjacency cache. Failure releases its state and writes no database
rows. Temporary indexes use the cached graph module's ID hash to validate edge
identity, locate outgoing edge lists and find visited nodes. The edge-ID lookup
scratch becomes the visited-node lookup after validation. Lists retain source
scan order; the selected parent chain is reversed for linear output construction.
Expected build/traversal work is linear in scanned edges and reached nodes;
adversarial hash collisions still consume the explicit work budget. Index scratch
uses additional payload bytes, so a graph may require a larger byte budget than
the former scan-based implementation. The payload ceiling and error codes remain
unchanged; no state is retained between calls.
The existing BFS/path helpers retain their traversal contracts. Their resident cache is byte-bounded separately; query work and temporary allocations remain outside that cache bound. The Rust adapter exposes this helper through typed `witness` requests. Existing `reach` and `traverse` operations keep their admission limits and output behavior.


The adapter `witness` operation requires a trusted `graph.edge_id` column mapping,
`g2/1`, the configured node/edge types, outgoing direction, source/target integers,
and explicit work/byte budgets. It reuses typed schema and scope admission in the
same read transaction. Missing or ineligible endpoints produce `missing_ids` and
an empty path. The native work/byte budgets apply to witness execution after SQL
admission; they do not bound SQLite admission work or process RSS.

MCP cancellation uses SQLite interruption during native witness execution and a
per-request progress hook for SQLite VM execution. A cancellation gate protects
queued and later requests from a stale interrupt. Cancelled operations remove the
hook and roll back any surviving read transaction before releasing the serialized
connection. CLI remains sequential JSONL with the same result/error envelopes;
its process can be stopped by the host. Legacy BFS native loops do not poll for
cancellation and may delay a cancelled legacy traversal until they return.
