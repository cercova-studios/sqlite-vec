# Filtered FTS query profiling

Compare the published eligibility query with a correlated `EXISTS` predicate on
public SciFact documents and real BGE embeddings. [REPORT.md](REPORT.md) records
the diagnosis, paired timings, correctness checks and selectivity tradeoffs.
The change affects the guide's keyword branch; ranking and defaults stay fixed.

## Reproduce

Run from the repository root on macOS arm64 with `cc`, `uv`, and the existing
`tests` Python project. Reuse the verified SciFact arrays in
`../2026-10-26-scifact-hybrid/.work/`. If they are absent, follow that archive's
[preparation instructions](../2026-10-26-scifact-hybrid/README.md#reproduce).
Preparation needs network access; cached benchmarks and archive checks do not.
Run Python with assertions enabled.

```sh
python3 benchmarks-ann/experiments/2026-10-26-scifact-hybrid/build.py --work benchmarks-ann/experiments/2026-11-01-filtered-fts/.work
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python benchmarks-ann/experiments/2026-11-01-filtered-fts/test_equivalence.py -v
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python benchmarks-ann/experiments/2026-11-01-filtered-fts/benchmark.py
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python benchmarks-ann/experiments/2026-11-01-filtered-fts/diagnose.py
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python benchmarks-ann/experiments/2026-11-01-filtered-fts/alternatives.py
```

Run the benchmark alone before profiling or running tests. It compares both SQL
variants on one library/database, alternates their order deterministically, and
uses the prior fixed 30-query sample, five rounds and branch budget 50. All 300
queries in each scope must reproduce the frozen candidate lists and metrics;
vector output is checked as a control. The baseline is the archived old SQL,
not an old binary. Warmups and profiler instrumentation are excluded from timings.
The slow baseline makes a full comparison take several minutes.

The builder freezes committed C sources and excludes dirty C edits. It records
the revision, compiler, header/source hashes and library hash. New runs write only
ignored `.work/`; they do not replace `recorded/`. The saved run uses `83206b4`.
Concurrent graph commits made later are outside that measured snapshot.

`diagnose.py` saves query plans, VM opcodes, CPU/wall measurements and SQLite
internal statement counts. It also attempts macOS native sampling of its own
worker and records failure when the OS prevents attachment. Sampling permission
is not required for the SQL diagnostics. `alternatives.py` checks whether
`PRAGMA optimize` or `ANALYZE` removes the baseline's repeated FTS scans.

## Verify the saved run

```sh
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python benchmarks-ann/experiments/2026-11-01-filtered-fts/verify.py
```

Verification needs the adjacent SciFact archive but no model downloads, arrays or
compiled library. It checks archive hashes, exact result tuples and relevance
metrics against the frozen reference, timing coverage, eligibility and regenerated
summaries. Corpus/model attribution, revisions and input hashes remain in the
[original archive](../2026-10-26-scifact-hybrid/README.md#dataset-and-model-attribution).
The new archive retains scripts, guide snapshots, plans and raw receipts, without
copying the dataset, model or binary. Host contention and SQLite versions can
change performance; these results do not establish a production latency target.
