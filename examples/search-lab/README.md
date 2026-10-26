# Local search tools

Search a small SQLite corpus, expand related nodes, and fetch evidence through
the same read-only operations in a JSON/JSONL CLI or a local stdio MCP server.
The optional Python adapter loads this fork's native extension. It runs no model
or embedding service.

## Run the fixture

Run these commands from the repository root on a machine with `uv`, Python 3.12
with SQLite extension loading, and a built `dist/vec0` library. `make loadable`
builds the library; use its actual `.dylib` or `.so` path. Managed Python 3.12
worked on macOS arm64; Apple's system Python did not expose extension loading.

```sh
uv sync --project examples/search-lab --locked
uv run --project examples/search-lab python examples/search-lab/fixture.py \
  /tmp/search-lab-demo --extension dist/vec0.dylib
uv run --project examples/search-lab python examples/search-lab/adapter.py \
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
uv run --project examples/search-lab python examples/search-lab/adapter.py \
  --config /tmp/search-lab-demo/config.json mcp
```

The server exposes the `retrieval` toolset as four named tools: `describe`,
`search`, `traverse` and `fetch`. Each takes `corpus` and an optional `args` object,
matching the CLI record without `op`. Tool discovery includes argument schemas.
For example, call `search` with:

```json
{"corpus":"demo","args":{"text":"deploy","vector":[0,0],"candidates":3,"limit":3}}
```

Results are in `structuredContent`; errors also set MCP `isError`. Stdout contains
only MCP protocol messages. The SDK calls the shared handlers directly on the
server's persistent SQLite connection. Calls execute serially in the server event
loop; a running native call blocks later calls. Stop the process to close it.

## Operations and results

| Operation | Arguments | Result |
|---|---|---|
| `describe` | No arguments | SQLite/extension versions, dimension/metric, ID domain, limits and capability flags |
| `search` | `mode`: `lexical`, `dense`, or default `hybrid`; lexical/hybrid require `text`; dense/hybrid require exactly one of `vector` or `embedding_id`; `candidates` default 50; `limit` default 10 | Stable IDs/source references, one-based branch ranks, vector L2 distance, keyword BM25, RRF score, branch counts and paging status |
| `traverse` | Unique `ids`; `depth` default 1; `limit` default 50 | Distinct nodes and minimum hops from any eligible seed, ordered by depth then ID; eligible seeds included at depth zero |
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
  "edges":"admitted_edges", "tenant":"ours", "as_of":100, "dimensions":2
}
```

Mappings accept simple SQL identifiers. Column names are fixed for this slice:

- Documents: `id INTEGER PRIMARY KEY`, `document_id`, `tenant`, `valid_from`,
  `valid_to`, `invalidated_at`, `source_id`, `body`, `start_ms`, `end_ms`.
- Vectors: `vec0(embedding float[N])`, with no additional options and the same
  rowids as documents. `N` must equal configured `dimensions`.
- FTS: an FTS5 table with document rowids and searchable text.
- Edges: a trusted table/view with integer `src` and `dst` columns.

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

SQLite opens the database with URI `mode=ro`. After native loading, extension
loading is disabled, `query_only` is enabled, and an authorizer rejects writes,
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
configured edge source, native BFS materializes results, and its cache is slot
bounded rather than byte bounded. SQL LIMIT does not stop that work. The adapter
does not advertise cancellation, deadlines, continuation or a hard native budget.
The SDK parses MCP messages before handler validation, so the request bound is not
a transport allocation cap. Protocol framing adds bytes to the bounded result
object. Trusted views may themselves be expensive; review them before admission.

## Verify and measure

```sh
uv run --project examples/search-lab --group dev pytest -q examples/search-lab/test_adapter.py
uv run --project examples/search-lab python examples/search-lab/measure.py \
  --config /tmp/search-lab-demo/config.json
```

Tests run the complete journey through real CLI and MCP subprocesses and compare
against independent direct SQL, including scalar distances, window-ranked RRF and
recursive minimum-hop reachability. [Recorded measurements and validation](experiments/2026-10-26-a0/VALIDATION.md)
describe the tested environment and remaining gaps. They do not claim an agent
task-success or production performance benefit.

The design borrows named parameterized tools/toolsets from
[Google MCP Toolbox](https://github.com/googleapis/mcp-toolbox#quick-start-custom-tools).
It uses the maintained [MCP Python SDK v1](https://py.sdk.modelcontextprotocol.io/v1/)
for local protocol handling. Toolbox is not a runtime dependency, and no assumption
about its pure-Go SQLite driver's native-extension loading is needed here.

Removing this optional directory and its architecture entry rolls back the adapter.
The adapter makes no database-format change; fixture creation is a separate write tool.
