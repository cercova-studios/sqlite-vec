# Embedded filtered-vector comparison

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

Compare upstream sqlite-vec, this fork's exact/HNSW paths, Zvec and standalone
LanceDB on identical float32 vectors, L2 queries and pre-search eligibility sets.
Measured on an Apple M5 Pro (18 cores, 48 GiB RAM), macOS arm64.
This is a local engine/SDK comparison; it bypasses the search-lab adapter and its
10,000-row admission limit. It changes no production engine, index default or CI.

## Workload and controls

- SciFact: all 5,183 pinned BGE-small 384-dimensional document vectors; 100 queries
  selected without replacement using seed 42 from the 300 pinned queries.
- SIFT: 50,000 training vectors sampled with seed 42 from the public ANN-Benchmarks
  file, sorted back to source order; the first 100 held-out test vectors, 128D.
  The full download hash matches the earlier HNSW experiment's pinned source.
- L2 top-10, one insertion permutation (seed 19), three warm timing rounds, one
  client, no embeddings or model inference in timings. Queries are shuffled per
  timing round. No confidence intervals or insertion/build variance are claimed.
- Eligibility: all, independently permuted approximately 10%/1% subsets, a 1%
  first-coordinate-order band, and an empty control. Masks are balanced integer
  buckets, shared across engines. The coordinate band is not a semantic label.
- The reference directly sums squared coordinate differences in float64 without
  BLAS. Strict recall resolves cutoff ties by ID; tie-aware recall requires all
  strictly closer neighbors and permits any remaining cutoff ties. Returning
  fewer than ten results reduces recall; empty-domain recall is conventionally 1
  and is excluded from matched-recall rankings.

SQLite uses an indexed ordinary scope table and rowid IN filtering. HNSW metadata
columns are unsupported, so no engine change was made to accommodate the harness.
Zvec uses scalar inverted indexes; LanceDB uses scalar B-trees and explicit
prefiltering. The all-documents case omits the filter on every engine. This fork
may switch selective HNSW queries to exact ranking of eligible rowids; the timings
measure the engine's query strategy, not exclusively graph traversal.

HNSW uses M=16 and construction ef=200. Query ef sweeps 32/128/512. LanceDB tests
unquantized IVF_HNSW_FLAT with one partition and with 64 partitions; the latter
uses respectively 8/32/64 probes. These are tested configurations, not claims of
optimal tuning. The recorded summary chooses the lowest observed median among
configurations reaching mean recall 0.95 or 0.99 for each case separately. That
selection is descriptive on these 100 queries, not a held-out tuning guarantee.

Each engine/dataset runs in a fresh sequential subprocess, in a fixed shuffled
engine order. Query timings include Python SDK/builder/result conversion overhead
and exclude vector preparation and correctness checks. Build time includes engine
import/setup, ingestion, scalar indexes, vector indexing and flush/commit as
applicable. Storage is post-build bytes under the database path, including retained
versions, not live-index size. RSS is whole-process high-water memory during build
and queries, including Python/import/input overhead; it is not native index RSS.
Fresh handles are reopened and checked in-process; this is not crash recovery.

Compute limits request one worker: Zvec query/optimization threads 1, Arrow CPU/I/O
threads 1, and OMP/OpenBLAS/Accelerate/Rayon/Lance compute limits 1. Lance I/O threads
are 2 after the retained one-thread scalar-index stalls. These settings do not pin
CPU affinity or make each engine's scheduler equivalent. File-backed databases
and warm queries do not establish cold-cache or concurrent performance.

## Replay on macOS arm64

Requires `cc`, `uv`, Python 3.12 and the repository's generated `sqlite-vec.h` and
bundled SQLite headers. `prepare.py` builds a separate committed-source fork
library, preserving the unrelated dirty HNSW edit and existing dist libraries.
Upstream sqlite-vec comes from its pinned wheel; its compiler flags are unknown,
so upstream/fork differences are not isolated source-level causal measurements.

```sh
experiment=benchmarks-ann/experiments/2026-11-03-engine-comparison
mkdir -p "$experiment/.work"
uv venv --python 3.12 "$experiment/.work/venv"
uv pip install --python "$experiment/.work/venv/bin/python" -r "$experiment/requirements.txt"
curl --fail --location https://ann-benchmarks.com/sift-128-euclidean.hdf5 \
  --output "$experiment/.work/sift.hdf5"
# Restore the pinned SciFact documents.npy and queries.npy in the earlier
# 2026-10-26-scifact-hybrid/.work directory, following that archive's replay.
"$experiment/.work/venv/bin/python" "$experiment/prepare.py"
"$experiment/.work/venv/bin/python" "$experiment/run.py"
"$experiment/.work/venv/bin/python" "$experiment/verify.py" --capture --write
```

Replays overwrite recorded receipts; use an isolated checkout to retain the frozen
measurement. Inputs, libraries and databases are ignored. Workers remove their
own databases on normal completion; controller timeouts/errors are failures.

Offline verification needs only Python's standard library:

```sh
python3 benchmarks-ann/experiments/2026-11-03-engine-comparison/verify.py
```

It validates hashes, returned-ID eligibility/uniqueness, strict/tie-aware recall,
exact-engine recall, counts, matched-target selection and summary regeneration.
The live workers additionally validate reported L2/squared-L2 units against
original vectors on every query, including timed queries, and check reopened
handles. Source hashes connect receipts to the measured harness.

## Retained compatibility evidence

`probe_threads.py` reproduces the LanceDB 0.41.0 scalar-index setup stall with one
I/O thread (15-second timeout at either one or two compute threads). Two I/O
threads complete both probes. This isolates a configuration dependency; no native
stack diagnosis was collected. `recorded/thread-probes.json` retains checkpoints
and outcomes. An earlier pilot also emitted per-query distance-column warnings;
those timings were discarded and final projections explicitly select `_distance`.

Primary API sources inspected during construction:
[Zvec query](https://zvec.org/en/docs/db/data-operations/query/),
[Zvec HNSW](https://zvec.org/en/docs/db/concepts/vector-index/hnsw-index/),
[LanceDB Python threading](https://github.com/lancedb/lancedb/blob/main/python/README.md).
Installed package signatures and documentation supplied the pinned API contracts.

Turso/libSQL, LadybugDB, zvec-grep, SQLite–Lance integration, hybrid ranking,
transactions/updates/deletes, crash recovery, million-scale data and end-to-end
CLI/MCP behavior remain separate comparisons.
