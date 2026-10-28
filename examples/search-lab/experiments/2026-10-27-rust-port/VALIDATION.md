# Rust adapter verification — October 27, 2026

The optional adapter now runs as a Rust binary. The CLI record format, four MCP
tools, configured scope, result fields, ordering, caps and read-only guarantees
retain the [current contract](../../README.md). Native BFS and the C extension
were not modified by this port.

## Checks

Executed from the repository root; Python commands used
`UV_CACHE_DIR=/tmp/uv-cache-muse`:

```sh
cargo test --release --offline --locked --manifest-path examples/search-lab/Cargo.toml
# 8 Rust tests passed
cargo build --release --offline --locked --manifest-path examples/search-lab/Cargo.toml
cargo fmt --check --manifest-path examples/search-lab/Cargo.toml
cargo clippy --offline --manifest-path examples/search-lab/Cargo.toml --all-targets -- -D warnings

uv run --offline --managed-python --project examples/search-lab --group dev pytest -q \
  examples/search-lab/test_adapter.py examples/search-lab/test_graph_semantics.py
# 65 tests passed

uv run --offline --managed-python --project tests pytest -q \
  tests/test-graph.py tests/test-graph-vector.py tests/test-loadable.py \
  -k 'graph or vec_rrf_scoped_hybrid_recipe'
# 22 passed, 63 deselected
```

The parity suite launches the Rust release binary through CLI and MCP subprocesses
and compares its journey against independent direct SQL. The 17 handwritten G0
cases run through both Rust transports as well as native SQL and the independent
Floyd-Warshall oracle. Existing seeded-graph, NULL, identity, multiplicity, optional
binding, scope and rollback checks remain active. A test runs the binary with no
Python or other command available on PATH.

Connection-internal checks moved from Python monkeypatches into Rust unit tests.
They verify authorizer rejection, `query_only` protection of TEMP, read-only file
writes after disabling both other layers, disabled native loading, rollback after
query errors, adjacency reuse/invalidation and a WAL commit during an operation.
The active operation sees its earlier read snapshot; the next operation sees the
commit. Actual database growth beyond 64 MiB is rejected at runtime and startup.
Malformed UTF-8, final records without a newline, signed-ID types and Unicode
byte accounting have explicit regression checks. Real CLI/MCP byte-cap tests remain.

Before removing the old runtime, 19 full result/error records were compared against
the implementation at `fee2050`: describe, all search modes/candidate depths, stored
embeddings, fetch, multi-seed expansion and malformed operations/IDs/FTS. Every
field matched except elapsed time and the accurately reported SQLite version.
This was a temporary migration check; the old runtime is not retained as a
compatibility shim. The durable tests use independent SQL and fixture expectations.

## Environment and measurements

Tested on macOS 27.0.1 arm64 with Rust 1.99.1. Cargo locks `rusqlite` 0.41.0,
`rmcp` 3.7.0 and Tokio 1.54.1. Bundled SQLite is 3.54.1; the independent Python
3.12.13 SQL oracle uses SQLite 3.53.1. Both load the existing `dist/vec0.dylib`
(`v0.1.10-alpha.4`). Hashes of the binary, native extension, fixture, lockfile and
measured runtime sources are in [measurements.json](measurements.json).

The updated measurement client runs 20 sequential warm samples per operation over
the same tiny fixture: 10 documents, 6 eligible records, two vector dimensions and
7 admitted edges. It makes no model or network calls.

| Operation | CLI p50 / p95, ms | MCP p50 / p95, ms |
|---|---:|---:|
| describe | 0.034 / 0.056 | 0.239 / 0.282 |
| search | 0.094 / 0.119 | 0.304 / 0.379 |
| traverse | 0.101 / 0.162 | 0.279 / 0.314 |
| fetch | 0.065 / 0.088 | 0.262 / 0.300 |

CLI spawn through its first describe response took 5.575 ms. MCP spawn and
initialization took 9.604 ms. The largest terminated child peak was 8,912,896 bytes;
this excludes the Python measurement client and is not combined process RSS.
The stripped release binary was 5,883,472 bytes. These are one-run plumbing
measurements on a warm OS cache, not production benchmarks or a controlled
before/after speedup claim.

Reproduce with a fresh fixture directory, then run:

```sh
uv run --project examples/search-lab --group dev python examples/search-lab/fixture.py \
  /tmp/search-lab-rust-check --extension dist/vec0.dylib
uv run --project examples/search-lab --group dev python examples/search-lab/measure.py \
  --config /tmp/search-lab-rust-check/config.json
```

## Cleanup and limits

Removed the Python runtime and its obsolete internal harness paths. Python remains
only for active test, fixture and measurement clients; its MCP dependency is in the
development group. The old A0/G0 execution receipts were consolidated into this
current receipt, and superseded measurement output was removed. G0 semantic data
and its independent oracle remain active. Production tracing was removed from the
normal dependency features; tests enable it only as a development dependency.
The C build, hybrid crate, unrelated HNSW edit, HANDOFF and other benchmark work
were left alone. No push or CI changes were made.

The official [Rust MCP SDK](https://github.com/modelcontextprotocol/rust-sdk) and
installed source supplied the stdio/server contract. Installed `rusqlite` source
and [extension guard documentation](https://docs.rs/rusqlite/0.41.0/rusqlite/struct.LoadExtensionGuard.html)
ground the loader and authorizer use. Graph-index coverage was checked; changed
or untracked Rust/test paths were reviewed directly rather than treating stale
metadata as proof of coverage.

Not checked: full integration/C suites, sanitizers, other platforms/toolchains,
large corpora, cold disk, concurrent MCP load or agent task quality. Native
work/memory budgets and cancellation remain unsupported. The current exact scalar
L2 and scoped distinct-node BFS semantics are preserved; no GraphIR compiler or
new graph capability was added.
