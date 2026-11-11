# Rust adapter keyword SQL profile

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

This experiment compares the adapter's materialized eligibility `IN` predicate
with correlated `EXISTS` using its locked `rusqlite` bundled SQLite. It reuses
all 5,183 public SciFact title/abstract records and all 300 frozen lexical
expressions, with full and independently scoped eligibility.

## Replay

From the repository root, restore the pinned SciFact `data.json` in the earlier
[experiment's ignored work directory](../2026-10-26-scifact-hybrid/README.md).
The earlier recorded keyword receipts are already committed. Preparation checks
the corpus SHA256, regenerates lexical expressions with FTS5 tokenization, and
writes ignored input data. The input hash and selected query IDs are recorded.

```sh
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python \
  benchmarks-ann/experiments/2026-11-02-adapter-fts/prepare.py
cargo run --release --locked --offline --manifest-path examples/search-lab/Cargo.toml \
  --example keyword-profile -- \
  benchmarks-ann/experiments/2026-11-02-adapter-fts/.work/input.json \
  /tmp/adapter-fts-replay.json
python3 benchmarks-ann/experiments/2026-11-02-adapter-fts/verify.py
```

The benchmark asserts exact same-build equality of ordered IDs and BM25 doubles
for both variants at all 600 query/scope points. Historical IDs must also match.
Archived cross-build BM25 values differed in their last floating-point bits on
initial comparison; the score-preservation assertion therefore compares both
predicates directly on the same bundled engine. No same-build tolerance is used.

Eight queries are selected by numeric ID from the prior frozen timing sample.
Each variant gets a warmup and three timed rounds per query/scope, with alternating
order. This yields 48 timed pairs. The rounds are descriptive, not confidence
intervals. Each timing includes statement preparation and complete retrieval.
SQLite VM-step counts and query plans provide SQL-level profiling; FTS internals
are not represented by VM-step counts. No native stack sampling was collected.

The database is in memory and includes the document ID/eligibility columns and
full FTS title/abstract content. Other document fields, vectors, adapter admission,
fusion, MCP progress hooks and transport overhead are excluded. This isolates the
keyword SQL rather than timing an end-to-end application. Controls cover six
eligible counts, common terms, a phrase and no-match text. These measurements do
not establish production, cold-cache, concurrency or other-platform performance.

Raw rows, timings, counters, plans and controls are in `recorded/results.json.gz`.
The verifier checks receipt shape, recorded hashes and summary regeneration.
The executable benchmark enforces candidate/score equality during replay.
