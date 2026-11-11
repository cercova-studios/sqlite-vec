# Bounded witness indexing

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

The native `graph_witness` path now uses temporary hash/list indexes instead of
repeated full scans. The selected parent chain is reversed for one-pass output.
All state remains in its existing byte-bounded arena. The implementation reuses
the graph module's ID hash and SQLite allocation, statement and interruption APIs;
it adds no persistent state, dependencies or SQLite planner changes.

Source scan order still breaks shortest-path ties. Source validation, signed
edge/node IDs, rejection of invalid or duplicate IDs, work/memory errors and
rollback visibility retain their contracts. Every index probe and initialization
visit consumes work and checks interruption. Hash collisions can still produce
quadratic work; the explicit work budget bounds it. Index scratch requires more
bytes than the earlier scan implementation for some inputs.

## Local before/after measurements

Baseline: committed `c956253`. Candidate: the same C/header snapshot with only
`sqlite-vec-graph.c` replaced. Both libraries compile with identical Apple clang
21 -O3/M1/NEON flags. The unrelated dirty HNSW helper is excluded. Generated
libraries live under ignored `.work/`; existing production sources stay intact.

Five alternating before/after process pairs query the same SQLite graph database.
Each process makes twelve calls across a 5,000-node chain, 5,000-node star,
seeded 1,000-node/4,000-edge random graph and 19,999 parallel edges. The first
call includes first-use SQL/library effects; subsequent calls are warm. Witnesses
never use an adjacency cache. Python SDK call timing excludes fixture generation,
loading the extension and correctness checks; it includes SQL preparation and
returned JSON copying. Work budget is one billion; payload budget is 16 MiB.

| Query | Before median ms | After median ms | Speedup |
|---|---:|---:|---:|
| Chain, 29 hops | 9.988 | 0.589 | 17.0x |
| Chain, 4,999 hops | 69.877 | 0.683 | 102.3x |
| Star, last outgoing edge | 19.059 | 0.493 | 38.7x |
| Random, 0 to 999 | 10.767 | 0.448 | 24.1x |
| Parallel edges, one hop | 156.345 | 1.670 | 93.6x |

All 120 calls pass a separate Python BFS distance/edge check. Every before/after
path hash also matches, including scan-order ties and unreachable/equal-endpoint
cases. The 4,999-hop chain exhausts 100,000 work units before the change and
succeeds within that budget afterward. Detailed results include all twelve
queries, source/library hashes and runtime versions.

Memory probes test a fixed power-of-two budget grid, 128 bytes through 16 MiB.
These are the smallest **tested successful budgets**, not exact allocation sizes
or RSS. Allocator rounding can reject arbitrary sizes, so no binary-search
monotonicity assumption is used. See `recorded/results.json` for every probe.

| Query | Before budget | After budget |
|---|---:|---:|
| Full chain witness | 512 KiB | 1 MiB |
| Random 0 to 999 | 128 KiB | 256 KiB |
| Parallel-edge witness | 512 KiB | 2 MiB |

The experiment ran on M5 Pro/macOS arm64, Python 3.12.13 with SQLite 3.53.1.
The final campaign ran serially after other test/build jobs finished. Earlier
overlapping pilot measurements were discarded. Five observations per query,
synthetic graph structure and one host do not establish production, concurrent,
cold-cache, million-edge or cross-platform performance. No external-engine
rerun or new engine ranking is implied.

## Verification and replay

From the repository root, on macOS arm64 with Python 3, `cc`, Git, generated
`sqlite-vec.h` and vendor SQLite headers:

```sh
python3 benchmarks-ann/experiments/2026-11-03-bounded-witness/bench.py
python3 benchmarks-ann/experiments/2026-11-03-bounded-witness/verify.py --write
python3 benchmarks-ann/experiments/2026-11-03-bounded-witness/verify.py
```

Replay replaces this experiment's `.work/graph.db`, libraries and recorded
results. Use a separate checkout to preserve frozen evidence. The baseline source
comes from Git; only the candidate graph source comes from the current tree.
`verify.py` needs no engine packages. It checks source hashes, witness lengths,
paired path equality, regenerated summaries and corruption rejection. Raw paths
are checked live; the archive retains their hashes and lengths, not every path.
It does not rerun native queries. `--write` updates archive hashes only after the
new receipts pass validation; it cannot establish performance or correctness by
itself.

The change was exercised through default native tests/snapshots, all 76 adapter/
CLI/MCP/G0 tests, graph oracle/rollback tests, native collision-heavy cancellation
and work-limit/retry tests, and the C unit harness. The unit harness now tries
every byte budget from 128 through 1,023 on a three-edge witness; successful
results must match exactly and failures must be `SQLITE_TOOBIG`.

AddressSanitizer/UndefinedBehaviorSanitizer unit execution passes. A no-HNSW build
passes a graph smoke check. The runtime loadable used for final application
checks is the isolated candidate library, excluding the unrelated HNSW edit.
Cancellation fixtures use deliberately colliding IDs so faster ordinary queries
cannot finish before interruption; queued-cancellation checks still require the
active request to succeed and the connection to remain reusable.

Hybrid Lance, full experimental-IVF loadable tests and other platforms were not
rerun. No push or CI changes.
