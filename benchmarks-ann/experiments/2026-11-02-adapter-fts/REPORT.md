# Bundled SQLite keyword comparison — 2026-11-02

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

Adopt correlated `EXISTS` in the adapter's keyword branch. On its bundled SQLite
3.54.1, all 600 SciFact query/scope comparisons return exactly equal ordered IDs
and BM25 doubles under `IN` and `EXISTS`. Historical reference IDs also match.
Only the membership predicate changes; filtering remains before the branch limit.
Vector SQL, score calculation, rank/rowid ties, candidate budgets and RRF stay fixed.

## Measurements

| Scope | IN p50 ms | EXISTS p50 ms | IN p95 ms | EXISTS p95 ms |
|---|---:|---:|---:|---:|
| Full: 5,183 eligible | 1778.01 | 3.74 | 2341.21 | 4.27 |
| Scoped: 3,091 eligible | 1082.96 | 2.62 | 1422.21 | 3.08 |

Each row pools 24 calls per variant: eight preselected queries, three paired
rounds, alternating variant order after warmups. These are descriptive percentiles
from a warm in-memory keyword benchmark on one development host. They exclude
admission, vectors, fusion, MCP cancellation hooks and transport. The eligibility
table has the adapter's ID/scope columns; other document fields are omitted.
See [replay and methodology](README.md) and [raw receipt](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-02-adapter-fts/recorded/results.json.gz).

The plan changes from FTS index `0:=M1` (rowid plus MATCH constraints) to `0:M1`,
followed by a correlated eligibility lookup with an automatic covering index and
Bloom filter. This is consistent with avoiding repeated FTS setup observed in
the [earlier guide profile](../2026-11-01-filtered-fts/REPORT.md); native term-index
calls were not instrumented in this experiment. Median VM steps actually increase:
161,514 → 208,090 full, and 110,188 → 161,101.5 scoped. VM steps exclude work inside
FTS callbacks and cannot alone explain wall time. No native stack profile was run.

## Selectivity tradeoff

For zero/one eligible document, matching common-term and phrase controls regress
by roughly 0.04–0.10 ms: baseline 0.09–0.11 ms versus 0.15–0.19 ms. At ten or more
eligible documents both matching controls improve. The no-match control improves
at every tested size. This is not a universal selectivity threshold.
The substantial broad-scope improvement justifies the rewrite with this documented
small-scope cost; no adaptive planner policy or new index is introduced.
All control tuples and BM25 values match exactly within the bundled build.

## Preservation and verification

The Rust regression compares both SQL variants across phrase/OR/NOT/no-match
queries, candidate limits, ties, signed rowid extremes, tenant/validity and
invalidation predicates. It exercises pending invalidation, changed FTS content,
savepoint rollback, full rollback, committed invalidation and an actual adapter
lexical call against the unchanged IN reference. Existing CLI/MCP hybrid parity,
read-only, snapshot and cancellation tests remain the end-to-end checks.

The initial archived-JSON comparison found last-bit BM25 differences across
builds/runtimes/parsers; its cause was not isolated. The retained
[counterexample](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-02-adapter-fts/recorded/initial-comparison.json) distinguishes that observation
from the exact, same-build IN/EXISTS assertion. No score tolerance is used for the
rewrite. No relevance or ranking improvement was inferred from latency. Models, schemas,
dependencies and ANN settings are unchanged. Cold-cache, concurrent clients, other platforms,
process RSS and complete application latency remain unverified.


Post-adoption verification: the release adapter builds; all 10 Rust unit tests
and all 76 adapter/CLI/MCP/G0 tests pass. Archive hash checks and summary
regeneration pass, and deliberate result-ID corruption is rejected. Python syntax,
Rust formatting and whitespace checks pass. Native C, Lance, full experimental-IVF,
sanitizers and other platforms were not rerun for this SQL-only change.
