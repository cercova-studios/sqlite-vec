# Local search tools

Search a small SQLite corpus, expand related nodes, and fetch evidence through
the same read-only operations in a JSON/JSONL CLI or a local stdio MCP server.
The optional Rust binary loads this fork's native extension and bundles SQLite.
It runs without a Python installation, model or embedding service. Python tools
generate the fixture and run independent tests.

For graph identity, minimum-hop behavior and reusable conformance fixtures, see
the [graph semantics contract](graph-semantics.md).

## Run the fixture

Run these commands from the repository root with Cargo, a C compiler, `uv`, and a
built `dist/vec0` library. `make loadable` builds the native extension; use its
actual `.dylib` or `.so` path. The adapter loads that library directly. The fixture
generator also needs a Python SQLite binding with extension loading; managed
Python 3.12 worked on macOS arm64. Apple's system Python lacked that capability.

```sh
cargo build --release --locked --manifest-path examples/search-lab/Cargo.toml
uv sync --project examples/search-lab --locked --group dev
uv run --project examples/search-lab python examples/search-lab/fixture.py \
  /tmp/search-lab-demo --extension dist/vec0.dylib
examples/search-lab/target/release/sqlite-vec-search-lab \
  --config /tmp/search-lab-demo/config.json cli <<'JSON'
{"op":"describe","corpus":"demo"}
{"op":"search","corpus":"demo","args":{"text":"deploy","vector":[0,0],"candidates":3,"limit":3}}
{"op":"traverse","corpus":"demo","args":{"ids":[1,2,3],"depth":2}}
{"op":"fetch","corpus":"demo","args":{"ids":[1,2,3,4]}}
JSON
```

The fixture creator requires a new directory without `fixture.db` or
`config.json`. Its text is original test material. Search returns IDs 1, 2 and 3;
expansion returns those seeds at depth zero and ID 4 at depth one. Fetch returns
the text, document/source IDs and millisecond time ranges.

The CLI reads one UTF-8 JSON object per line, including a single final object
without a newline. It returns one result or error per record and continues after
record errors. Exit status is 0 for success, 1 if any record failed, or 2 for a
startup failure. Startup errors go to stderr. It retains one connection until EOF.

## MCP stdio

Configure a local MCP host to start this command, using absolute paths in the host:

```sh
examples/search-lab/target/release/sqlite-vec-search-lab \
  --config /tmp/search-lab-demo/config.json mcp
```

The server exposes the `retrieval` toolset as six named tools: `describe`,
`search`, `traverse`, `reach`, `fetch` and `witness`. Each takes `corpus` and an optional `args` object,
matching the CLI record without `op`. Tool discovery includes argument schemas.
For example, call `search` with:

```json
{"corpus":"demo","args":{"text":"deploy","vector":[0,0],"candidates":3,"limit":3}}
```

Results are in `structuredContent`; errors also set MCP `isError`. Stdout contains
only MCP protocol messages. The SDK calls the shared handlers directly on the
server's persistent SQLite connection. Calls use one serialized connection on a blocking worker. The protocol loop remains responsive to cancellation and tool discovery. MCP cancellation interrupts a running witness; queued cancellation does not interrupt another request. Stop the CLI process to cancel CLI work.

## Operations and results

| Operation | Arguments | Result |
|---|---|---|
| `describe` | No arguments | SQLite/extension versions, dimension/metric, ID domain, limits and capability flags |
| `search` | `mode`: `lexical`, `dense`, or default `hybrid`; lexical/hybrid require `text`; dense/hybrid require exactly one of `vector` or `embedding_id`; `candidates` default 50; `limit` default 10 | Stable IDs/source references, one-based branch ranks, vector L2 distance, keyword BM25, RRF score, branch counts and paging status |
| `traverse` | Unique `ids`; `depth` default 1; `limit` default 50 | Distinct nodes and minimum hops from any eligible seed, ordered by depth then ID; eligible seeds included at depth zero |
| `reach` | Typed `g1/1` request described below | Validated minimum-hop reachability in the configured node and edge domain |
| `witness` | Typed `g2/1` request with explicit work and byte budgets, described below | One shortest path with stable incoming edge IDs |
| `fetch` | Unique `ids` | Evidence in requested ID order, plus explicit `missing_ids` for absent or ineligible IDs |

`text` uses FTS5 query syntax, including phrases and Boolean operators. Malformed
FTS expressions return `database_error`. Supplied vectors must match configured
dimensions and contain finite numeric coordinates with magnitude at most `1e10`.
Stored embedding references use an eligible vector row's signed 64-bit ID. Clients
must preserve integer precision when serializing IDs.

Search computes exact scalar L2 over eligible flat vectors. It ranks each branch
by score ascending, then ID, **before** its candidate limit. This also resolves
equal-distance cutoff ties. Every mode reports RRF with constant 60 over its active
branches; higher RRF is better. Lower L2 and lower BM25 are better. Missing branch
ranks/distances are JSON null. Final order is RRF descending, then ID ascending.
This exact scan deliberately trades ANN speed for deterministic behavior on the
admitted small corpus. The adapter accepts no ANN table configuration.

`complete` describes the declared result domain, and becomes false when `truncated`
is true. For search that domain is the **bounded candidate union**, not every
matching record in the corpus. `boundary_hit` means a branch filled its candidate
allowance; it does not prove exhaustion. Exact distances and `approximation:false`
do not establish semantic relevance or exhaustive lexical/hybrid recall. Traversal
computes `total_reachable` before applying its output row limit. The envelope
records the operation, scope, elapsed handler time and termination reason.

## Configure a corpus

Treat the config, database and extension as trusted host inputs. The fixture
generator writes a complete config example. Relative filesystem paths resolve
against the config directory. A config names `database`, `extension`, and one to
four named `corpora`; each corpus supplies:

```json
{
  "documents":"documents", "vectors":"vectors", "fts":"lexical",
  "edges":"admitted_edges", "tenant":"ours", "as_of":100, "dimensions":2,
  "graph":{"node_type":"Function","edge_type":"CALLS"}
}
```

Mappings accept simple SQL identifiers. Column names are fixed for this slice:

- Documents: `id INTEGER PRIMARY KEY`, `document_id`, `tenant`, `valid_from`,
  `valid_to`, `invalidated_at`, `source_id`, `body`, `start_ms`, `end_ms`.
- Vectors: `vec0(embedding float[N])`, with no additional options and the same
  rowids as documents. `N` must equal configured `dimensions`.
- FTS: an FTS5 table with document rowids and searchable text.
- Edges: a trusted table/view with integer `src` and `dst` columns.

The optional `graph` mapping enables `reach` for one logical node type and one
edge type. The host is responsible for mapping that edge type to a source already
filtered to the intended relationship and scope. Existing configurations without
`graph` continue to support the other operations. `describe` returns the
configured types, or null when absent.

For example, the following CLI record requests distinct nodes whose minimum hop
distance from either eligible anchor is at most two:

```json
{"op":"reach","corpus":"demo","args":{"version":"g1/1","node_type":"Function","edge_type":"CALLS","direction":"out","anchors":[1,5],"max_depth":2,"limit":50}}
```

All seven arguments are required. Anchors are unique signed 64-bit IDs (1–64),
`max_depth` is 0–8, and `limit` is 1–100. Only `g1/1` and outgoing traversal are
supported. Unknown types, versions or directions return `unsupported`; unknown
fields and malformed values return `invalid_input`. The binder checks the node
table's single `INTEGER PRIMARY KEY` and eligibility columns, plus edge `src` and
`dst` columns, in the same read transaction used for traversal. An incompatible
mapping returns `schema_error`. Edge endpoint values and scope are validated by
the existing traversal handler. A missing or ineligible anchor is listed in
`missing_ids`; an eligible isolated anchor appears at depth zero.

Results preserve the existing depth/ID ordering, truncation and completeness
fields and add `version`, `node_type` and `edge_type`. This operation does not
accept arbitrary SQL, lower hop bounds, path predicates or relationship witnesses.
Minimum distance is not the same as the existence of a longer path.

The host fixes tenant and Unix-second `as_of`. Eligibility requires
`invalidated_at IS NULL`, `valid_from <= as_of`, and `valid_to > as_of` or NULL.
Every search branch applies it before its candidate limit. Fetch and embedding
references use the same predicate. Traversal validates that **all** configured
edge endpoints are eligible before native BFS runs. Configure a scoped edge view,
as in the fixture; an unscoped source is rejected rather than traversed and filtered
afterward. Unknown anchors are listed as missing instead of inventing nodes.
Traversal returns node reachability, not edge witnesses or path enumeration.

Scope is retrieval eligibility, not authentication. All callers of one process can
select any configured corpus. The local host must control which users/processes
can launch it and protect config/database files. There is no remote endpoint or
per-caller authorization service. Requests cannot select files, extensions or SQL.

SQLite opens the database with `SQLITE_OPEN_READ_ONLY`. After native loading, a
scoped guard disables further loading, `query_only` is enabled, and an authorizer rejects writes,
ATTACH, unsafe functions and configuration changes. The tests exercise each layer,
including a disk write after removing the other two protections. Database errors
remain structured errors; their messages may expose trusted schema names.

Each operation establishes one read transaction across admission checks and its
queries. Four calls do not share a snapshot. Parity tests use an unchanged fixture;
a separate WAL test commits evidence during an operation and checks that only the
next operation sees it. A persistent connection can retain native adjacency caches;
it does not freeze the database revision.

## Limits and failure handling

| Admission/output bound | Maximum |
|---|---:|
| CLI input record / canonical operation request | 65,536 bytes |
| Canonical result object | 262,144 bytes |
| Rows per document/vector/FTS mapping | 10,000 |
| Database logical pages | 64 MiB |
| Configured edge-source rows | 20,000 |
| Dimensions / branch candidates | 4,096 each |
| Search/traversal output rows | 100 |
| Input IDs / traversal depth | 64 / 8 |

`describe` reports these limits. Oversized output returns an `output_limit` error
with no partial evidence; request fewer rows. Missing fetch IDs remain explicit in
a successful result. Invalid requests, missing embeddings, rejected graph scope,
admission exhaustion and SQLite failures have distinct error codes.

These bounds restrict admitted workloads and returned data. They are not hard
process-memory or elapsed-work guarantees. Graph construction scans the entire
configured edge source and native BFS materializes results. Its resident cache
is bounded to eight graphs and 16 MiB; SQL LIMIT does not bound construction or
traversal work. The bounded `witness` operation supplies native payload/work
limits and MCP cancellation as described below. Deadlines and continuation remain
unsupported.
The SDK parses MCP messages before handler validation, so the request bound is not
a transport allocation cap. Protocol framing adds bytes to the bounded result
object. Trusted views may themselves be expensive; review them before admission.

## Verify and measure

```sh
cargo test --release --locked --manifest-path examples/search-lab/Cargo.toml
uv run --project examples/search-lab --group dev pytest -q \
  examples/search-lab/test_adapter.py examples/search-lab/test_graph_semantics.py
uv run --project examples/search-lab python examples/search-lab/measure.py \
  --config /tmp/search-lab-demo/config.json
```

Tests run the complete journey through real CLI and MCP subprocesses and compare
against independent direct SQL, including scalar distances, window-ranked RRF and
recursive minimum-hop reachability. Rust unit tests exercise the connection's
read-only layers, cache invalidation, failed-operation rollback and WAL snapshots.
[Recorded measurements and validation](experiments/2026-10-27-rust-port/VALIDATION.md)
describe the tested environment and remaining gaps. They do not claim an agent
task-success or production performance benefit.

The design borrows named parameterized tools/toolsets from
[Google MCP Toolbox](https://github.com/googleapis/mcp-toolbox#quick-start-custom-tools).
It uses the official [MCP Rust SDK](https://github.com/modelcontextprotocol/rust-sdk)
for local protocol handling. The Python MCP SDK is a test-client dependency only.
Toolbox is not a runtime dependency, and no assumption
about its pure-Go SQLite driver's native-extension loading is needed here.

Removing this optional directory and its architecture entry rolls back the adapter.
The adapter makes no database-format change; fixture creation is a separate write tool.


## Bounded graph evidence

To enable `witness`, add a stable ID column to the trusted graph mapping:
`"graph":{"node_type":"Function","edge_type":"CALLS","edge_id":"id"}`.
The selected edge table/view must expose that column, plus `src` and `dst`.
The host owns ID stability across updates and database maintenance. IDs must be
unique non-NULL signed integers within the configured corpus. Parallel edges
need distinct IDs. The existing demo edge view has no ID column; configure an
ID-bearing source before using this operation.

```json
{"op":"witness","corpus":"demo","args":{"version":"g2/1","node_type":"Function","edge_type":"CALLS","direction":"out","source":1,"target":4,"work":1000000,"bytes":65536}}
```

All eight arguments are required. `work` is 1–1,000,000,000 native units;
`bytes` is 128–67,108,864. Source and target are signed 64-bit IDs. The adapter
validates configured types, schema and endpoint eligibility in one read snapshot.
The result's `path` contains `[node, incoming_edge_id]` pairs; the first edge is
null. It returns one shortest directed path, any shortest tie in scan order.
Disconnected or missing endpoints return an empty path; missing/ineligible
endpoints also appear in `missing_ids`. Equal eligible endpoints return one node.

Budget exhaustion returns `work_limit` or `memory_limit`; MCP interruption returns
`cancelled` when a response is delivered. No partial witness is returned. A client
may stop waiting after sending MCP cancellation; the server still cleans up the
operation before reusing its connection. Invalid edge IDs return `database_error`.

The byte budget covers native edge rows, queue/parents and JSON construction.
SQLite admission/VM/preparation memory, allocator bookkeeping and returned-text
copies are excluded. Native work units cover scanned rows and loop visits or
comparisons after admission. These are not process memory or wall-clock limits.
Temporary indexes speed edge validation and traversal within the same payload
budget. Hash collisions can still consume quadratic work; every probe checks the
work limit and cancellation. Index scratch may require a larger byte budget than
earlier scan-based builds. Choose budgets suitable for the corpus.
`describe.witness_limits` reports these operation-specific capabilities. Existing
`reach`/`traverse` requests retain their prior budgets and behavior.

Separately, the native BFS cache retains at most 16 MiB across eight entries.
Oversized graphs are used uncached. Build scratch and cursor output are outside
that resident bound; `graph_cache_bytes()` reports current retained bytes.
