# A0 adapter validation — October 26, 2026

The CLI and local stdio MCP adapter completed describe → hybrid search → graph
expansion → evidence fetch against independent equivalent SQL. This receipt applies
to the optional adapter, not GraphIR or a production search service.

## Environment and artifacts

- macOS 27.0.1 arm64; managed CPython 3.12.13; SQLite 3.53.1; MCP SDK 1.30.0.
- Native artifact: `dist/vec0.dylib`, extension version `v0.1.10-alpha.4`.
  Its SHA256 is in [measurements.json](measurements.json). No C rebuild was needed
  for this host-only change; the receipt identifies the tested binary directly.
- Source checkout started at `c85e736`; documentation commits advanced HEAD to
  `52d1f14` during this task. The pre-existing `hnsw_allowed` helper edit was
  preserved and excluded from this change. HNSW alpha remains 1.0.
- Fixture: 10 original-text documents, 6 eligible at tenant `ours` / Unix second
  100, two float32 dimensions, and 7 admitted directed edges. It includes a
  diamond, cycle, parallel edges, self-loop, isolated node, cross-tenant records,
  validity boundaries and invalidation. `fixture.py` regenerates it without any
  downloaded corpus. Binary/config artifacts stay outside tracked source.
- Raw latency samples, environment, fixture/extension hashes and measured-script
  hashes are retained in [measurements.json](measurements.json).

## Checks executed

From the repository root, using `UV_CACHE_DIR=/tmp/uv-cache-muse`:

```sh
uv run --offline --managed-python --project examples/search-lab --group dev \
  pytest -q examples/search-lab/test_adapter.py
# 36 passed

uv run --offline --managed-python --project tests pytest -q \
  tests/test-graph.py tests/test-graph-vector.py tests/test-loadable.py \
  -k 'graph or vec_rrf_scoped_hybrid_recipe'
# 22 passed, 63 deselected

uv run --offline --managed-python --project examples/search-lab \
  python examples/search-lab/fixture.py /tmp/sqlite-vec-a0.bBLhVP \
  --extension dist/vec0.dylib
uv run --offline --managed-python --project examples/search-lab \
  python examples/search-lab/measure.py --config /tmp/sqlite-vec-a0.bBLhVP/config.json
```

The tests launch real CLI and MCP subprocesses on unchanged fixtures. Direct SQL
independently computes scalar distances, window-ranked RRF, minimum hops with a
recursive CTE and native BFS, and selected evidence fields. Exact assertions check
IDs, order, ties, ranks, scores/distances, hops and evidence across both transports.
They do not derive expected results from the adapter's SQL templates.

The first run found a real cutoff-tie mismatch between native KNN and exhaustive
SQL. Sorting a bounded KNN result cannot recover an excluded equal-distance row.
The adapter now uses exact scalar L2 with `(distance, id)` ordering before its
candidate limit. A regression checks the tied boundary explicitly. This is a
small-corpus policy, not an ANN speed improvement.

Additional checks cover malformed JSON/records, structured transport errors,
invalid dimensions/coordinates, unknown tools/corpora/IDs, empty branches,
tenant/validity isolation, duplicate and excessive IDs, excessive candidate/depth
requests, actual CLI/MCP input/output caps, output truncation, database failures,
graph/corpus admission and startup failure without creating a missing database.
Write tests reject disk writes, TEMP writes, ATTACH, `load_extension` and disabling
`query_only`. Removing the authorizer and `query_only` still leaves URI `mode=ro`
rejecting disk writes.

A WAL test commits evidence after an operation establishes its read snapshot:
that operation sees the old evidence and the next sees the new evidence. A separate
graph test observes a later committed edge on the same persistent connection.
SQL tracing records one native adjacency-build SELECT across repeated unchanged
traversals. This verifies cache reuse on the fixture, not a general cache-hit rate.

Graph-index coverage was checked for the adapter scripts and relied-on existing
paths. Native graph macro lines 596–598 have partial parser coverage and were read
directly. Source inspection and runtime tests ground the graph claims. The optional
semantic skill router failed to initialize its local Metal backend; the session
catalog supplied relevant skills. Shared source-library recall was unavailable
(`Operation not permitted`); inspected source links and findings remain here.

## Measurements

One run measured 20 warm calls per operation/transport after a first call. Calls
were sequential, without model, network or embedding work. The OS cache was warm.
The measurement runner asserts successful responses and one observed native graph
build. Values below are milliseconds, rounded; raw samples remain in the JSON.

| Operation | Handler p50 / p95 | CLI p50 / p95 | MCP p50 / p95 |
|---|---:|---:|---:|
| describe | 0.024 / 0.046 | 0.061 / 0.126 | 0.583 / 0.690 |
| search | 0.093 / 0.107 | 0.108 / 0.130 | 0.682 / 0.828 |
| traverse | 0.050 / 0.063 | 0.066 / 0.119 | 0.567 / 0.644 |
| fetch | 0.041 / 0.043 | 0.054 / 0.057 | 0.559 / 0.591 |

Opening/loading the handler connection took 0.862 ms. CLI spawn through its first
describe response took 32.585 ms. MCP spawn and initialization took 169.642 ms;
its first describe call then took 1.888 ms. First graph traversal took 0.211 ms
in-process, 0.246 ms through CLI and 0.753 ms through MCP. These first calls measure
fresh process/connection/cache paths, not cold-disk performance.

`getrusage` reported a 60,833,792-byte parent-process peak, including the MCP test
client, and a 62,521,344-byte largest terminated child peak across CLI/MCP. These
are process high-water marks, not per-query allocations or simultaneous combined
RSS. No isolated graph-cache byte accounting was measured.

## Scope and sources

The runnable slice stops at A0. No GraphIR compiler, Cypher parser, native graph
budget, model call, agent orchestrator, push or CI workflow was added. The source
chat's HANDOFF and benchmark archives were not edited.

Read the local roadmap's A0 contract, curriculum and Luxir comparison. Runtime
choices were checked against the installed SDK API and these primary references:

- [MCP Python SDK v1 maintenance documentation](https://py.sdk.modelcontextprotocol.io/v1/)
  and installed `mcp.server.lowlevel.Server.call_tool` / stdio implementation:
  named schemas and structured content use the maintained 1.30 SDK directly.
- [Google MCP Toolbox custom tools](https://github.com/googleapis/mcp-toolbox#quick-start-custom-tools):
  named parameterized tools grouped as a retrieval toolset, without adopting its
  database platform or assuming native-extension loading in its SQLite driver.
- Local `site/guides/hybrid-search.md`: eligibility inside each candidate branch,
  one-based RRF ranks, null missing ranks and explicit source references.
- Local `sqlite-vec-graph.c`: distinct-node BFS, whole-source CSR construction,
  materialized output and connection cache behavior.

Not checked: full C/integration suites, sanitizer builds, other OS/Python versions,
ANN recall, large corpora, cold disk, concurrent MCP load, native cancellation,
hard memory/work bounds, production authorization, or held-out agent task success,
repairs and token use. This fixture establishes correctness of the supported
plumbing; it cannot establish semantic retrieval quality or a general speedup.
