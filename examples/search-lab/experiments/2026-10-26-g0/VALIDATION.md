# G0 graph semantics validation — October 26, 2026

G0 adds a [semantic contract](../../graph-semantics.md), language-neutral
[fixture data](../../graph-fixtures.json), an independent test oracle and
conformance tests. Native graph implementation and adapter handlers are unchanged.

## Verified scope

The six fixture graphs contain 17 handwritten expected traversals. The test-only
oracle uses Floyd-Warshall all-pairs relaxation rather than native queue-based BFS.
Every handwritten expectation matches both implementations. Tests also compare
all selected anchors/depths, reversed directions, undirected union views, legal
shortest node sequences, and 24 deterministic tiny graphs under edge reordering,
duplication and added self-loops.

SQL checks preserve graph/type node identity, graph-scoped edge identity, bag
multiplicity, repeated-variable equality, nullable properties and optional-binding
counts. The shortcut counterexample distinguishes minimum distance from existence
of a longer binding. NULL validation-order cases preserve native behavior even at
depth zero. The scope fixture rejects paths through an ineligible intermediary;
adapter checks cover multiple seeds, isolated/absent/ineligible anchors, stable
ordering, output limits and completeness. Transaction checks compare each state
with the oracle across own writes, savepoint rollback, full rollback and commit.

Two deliberate in-memory AST mutations calibrated the handwritten oracle cases:

| Wrong oracle semantics | Handwritten cases that rejected the mutation |
|---|---:|
| Exclusive maximum depth (`hops < max_depth`) | 6 of 17 |
| Reversed edge direction | 11 of 17 |

The unmodified oracle matched all 17. These were targeted checks of the oracle's
expectations, not a production mutation-coverage measurement. The mutations were
never written into the working tree.

## Commands and results

Executed from the repository root with `UV_CACHE_DIR=/tmp/uv-cache-muse`:

```sh
uv run --offline --managed-python --project examples/search-lab --group dev \
  pytest -q examples/search-lab/test_graph_semantics.py examples/search-lab/test_adapter.py
# 63 passed: 27 G0 tests and 36 existing adapter tests

uv run --offline --managed-python --project tests pytest -q \
  tests/test-graph.py tests/test-graph-vector.py tests/test-loadable.py \
  -k 'graph or vec_rrf_scoped_hybrid_recipe'
# 22 passed, 63 deselected
```

The existing adapter suite includes real CLI/MCP journeys and snapshot checks.
The G0 checks run native SQL and adapter handlers against the oracle; they add no
new transport implementation. No performance benchmark was repeated because
neither runtime implementation changed.

Environment: macOS 27.0.1 arm64, managed CPython 3.12.13, SQLite 3.53.1. The tested
existing `dist/vec0.dylib` SHA256 was
`437f5a976eb409444621fb7f66b3436c7714068855cc5ed0065ad817d7eefcbd`.
No C rebuild was needed for this test/documentation change. The entry commit was
`d8800a2`; concurrent independent benchmark work reached `61fa78e` before commit.
Its files, HANDOFF, `.claude/` worktrees and the pre-existing HNSW helper edit were
left untouched. HNSW alpha remains 1.0.

Graph-index coverage was checked for relied-on graph source, adapter/fixture code
and new conformance files. Native graph macro lines 596–598 have partial parser
coverage and were read directly. Current source and execution results ground the
contract; graph-index coverage is a discovery aid, not a completeness proof.

## Host-language decision

The user selected a Rust-or-Go direction for the adapter during G0. Rust was chosen
to align with the existing `lance/Cargo.toml` use of `rusqlite` and Tokio. The current
adapter remains Python in this commit. Its port must retain the tested handlers,
read-only controls, transport parity and native-extension loading; the G0 JSON
fixtures and test-only oracle provide a baseline without prescribing a runtime.
The host choice does not add Rust or Go dependencies to the C core.

Not run: full C/integration suites, sanitizers, other platforms, large graphs,
performance comparisons, native cancellation or hard memory/work limits. G0 does
not implement G1 binding/compiler semantics, a Rust port, Cypher conformance or
path enumeration. No push or CI changes were made.
