# Filtered FTS profiling report — 2026-11-01

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

The scoped recipe repeatedly restarted the FTS query for eligible rowids. Changing
only the keyword eligibility predicate to correlated `EXISTS` removes that work
on the measured SQLite build. All returned tuples and relevance metrics match the
prior evaluation. Median keyword latency falls from seconds to milliseconds.

## Diagnosis and change

SciFact query `3` is a representative long OR expression. Before profiling,
wall/CPU time was 1.732/1.729 seconds: execution was predominantly on CPU.
The baseline plan is `SCAN fts_docs VIRTUAL TABLE INDEX 0:=M1`. Its VM opcodes
loop over the eligible-ID list around `VFilter`, restarting FTS for each ID.
SQLite's internal trace records 787,512 term-index lookups and 5,179 document-size
lookups. This is repeated FTS setup, not a slow vector-distance or fusion kernel.

With correlated `EXISTS`, the plan is `INDEX 0:M1`, followed by an indexed
eligibility lookup with a Bloom filter. The trace records 152 term-index lookups
and the same 5,179 document-size lookups. Plain wall/CPU time is 4.826/4.789 ms.
Every returned ID and BM25 score is identical. The complete plans, VM opcodes,
counts and rows are saved in [diagnostic.json](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-01-filtered-fts/recorded/diagnostic.json).

The guide changes:

```sql
-- Before, in the keyword branch:
and rowid in (select id from eligible)
-- After:
and exists (select 1 from eligible where eligible.id = fts_docs.rowid)
```

The vector branch retains its `rowid IN` prefilter. Eligibility remains inside
both branches before their limits. BM25, rowid tie order, RRF weights/constant 60,
branch budget 50, final top 10, distances, NULL branch ranks and provenance remain
unchanged. HNSW alpha stays 1.0. Integer primary-key IDs make membership equivalent
for the documented schema; this is not a claim about arbitrary affinity/collation
changes or arbitrary SQL rewrites. Inputs use valid FTS5 query syntax.

A unary-plus diagnostic also removes the rowid-driven plan, but is not shipped.
It is a planner hint with affinity consequences; correlated `EXISTS` expresses
the intended membership check directly. [SQLite's optimizer documentation](https://sqlite.org/optoverview.html#disqualifying_where_clause_terms_using_unary_)
describes that unary-plus limitation. In the independent statistics control,
`PRAGMA optimize` and `ANALYZE` leave the baseline's `=M1` plan intact:
1.685 seconds before, 1.731/1.712 seconds afterward versus 5.090 ms for `EXISTS`.
These are diagnostic single calls, not the performance comparison below.

Native macOS `sample` returned 255 when attaching to the experiment's own worker;
[sample-status.json](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-01-filtered-fts/recorded/sample-status.json) retains the denial. No native
stack or flame graph was obtained. SQL tracing adds overhead: the baseline takes
2.016 seconds instrumented versus 1.732 plain (+16.4%); `EXISTS` takes 6.659 ms
versus 4.826 ms (+38.0%). Progress counts are coarse lower bounds at 10,000 VM
instructions per callback, not native instruction counts. Instrumented calls do
not enter the benchmark timings.

## Paired comparison

All 5,183 public SciFact abstracts and 300 labelled test queries use the previously
pinned, locally generated BGE-small-en-v1.5 embeddings. Full scope admits all
documents; the independent validity/tenant overlay admits 3,091. Inputs are checked
against the previous data/array hashes. No model, lexical expression, index,
ranking or relevance-label tuning occurred.

Diagnostics use an in-memory database; the paired comparison uses a file-backed
database with the same schema and content. The timing sample is the prior seed-42
set of 30 queries, selected before this change. Each variant receives a warmup;
each query/method/context has five timed
pairs with deterministically shuffled order. Both variants run sequentially on
the same database and freshly compiled library. There are 150 pairs per row,
600 pairs total and 1,200 timed SQL calls. Percentiles pool the 150 calls per
variant. Geometric speedup averages the paired baseline/candidate ratios; it is
not a ratio of medians or a concurrent throughput measurement.

| Context | Method | Old p50 ms | New p50 ms | Old p95 ms | New p95 ms | Paired geometric speedup |
|---|---|---:|---:|---:|---:|---:|
| full | keyword | 1255.75 | 4.00 | 2576.58 | 6.40 | 294.5x |
| full | hybrid | 1303.52 | 6.71 | 2614.57 | 9.08 | 175.6x |
| scoped | keyword | 752.21 | 2.78 | 1498.64 | 4.21 | 251.5x |
| scoped | hybrid | 754.11 | 5.08 | 1505.99 | 6.55 | 137.7x |

[Raw comparison](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-01-filtered-fts/recorded/comparison.json.gz) retains result tuples, metrics,
paired wall/CPU measurements, order, query IDs, controls and query plans.
[summary.json](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-01-filtered-fts/recorded/summary.json) is regenerated by the verifier. These are
warm-cache, sequential observations on an unpinned development host; concurrent
development was not isolated. They do not predict production tails, concurrency,
I/O saturation, memory usage or other SQLite versions.

## Selectivity tradeoff

Two fixed queries (`1` and `3`) use 0/1/10/100/1,000/5,183 eligible documents.
Each cell is the median of three plain calls in milliseconds. Membership and
ranking match exactly at every size.

| Eligible docs | Query 1 old | Query 1 new | Query 3 old | Query 3 new |
|---:|---:|---:|---:|---:|
| 0 | 0.102 | 0.196 | 0.101 | 0.591 |
| 1 | 0.170 | 0.250 | 0.405 | 0.804 |
| 10 | 0.387 | 0.246 | 3.038 | 0.836 |
| 100 | 3.360 | 0.284 | 30.373 | 1.003 |
| 1000 | 37.858 | 0.583 | 300.946 | 1.682 |
| 5183 | 228.649 | 1.841 | 1755.513 | 4.785 |

The new plan still visits full-text matches when almost no document is eligible.
At zero/one eligible document it costs up to about 0.5 ms more in these controls.
At ten or more it wins for both control queries. This is a measured tradeoff,
not evidence of a universal crossover size. Keep tiny-scope production workloads
in future measurements; do not add adaptive SQL or change retrieval budgets from
these two queries alone.

## Correctness and verification

All 300 queries in both contexts reproduce the frozen keyword candidate lists
(up to 50) and hybrid output tuples exactly: 1,200 quality points. IDs/order,
BM25/fused scores, branch ranks, distances and source IDs match. All 600 vector
candidate lists are also compared as unchanged controls. No ineligible result
appears. Recall@10, nDCG@10 and reciprocal rank remain identical, including NULL
metrics when no positive judgment survives scope. The prior conclusion that
hybrid shows no demonstrated gain over vector remains unchanged.

Two executable equivalence tests cover tenant/time/invalidation combinations,
FTS OR/phrase/prefix/NOT/NEAR queries, absent tokens, branch limits 1/2/50,
cutoff ties, signed int64 boundary rowids, disabled automatic indexes, own writes,
savepoint/full rollback, committed updates, reopen and ANALYZE. The published
recipe's three existing vec_rrf tests also pass, including keyword-only NULL
provenance and empty scope. Those extra assertions were picked up by concurrent
commit `5415af2`; this commit preserves them without restaging that file.

The offline verifier checks hashes, exact reference equality, eligibility,
regenerated metrics, complete quality/timing coverage, profile row equality and
timing summaries. Corruption rejection is exercised separately. Full engine,
Rust/Lance, sanitizer and cross-platform suites were not rerun for this SQL-guide
change. No engine source or dependency changes are required.

## Provenance and next work

Measured build: `83206b49758581187ea68ab8ec0e78dcbe036d85`, Apple clang 21.0.0,
`-O3 -mcpu=apple-m1 -DSQLITE_VEC_ENABLE_NEON`, macOS 27.2 arm64, Python 3.12.13,
SQLite 3.53.1 and NumPy 2.1.3. [build-manifest.json](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-01-filtered-fts/recorded/build-manifest.json)
records all compiler flags, library/source hashes and the excluded unrelated HNSW
working-tree edit. C/header hashes equal the prior SciFact snapshot despite its
older commit ID; the baseline here is newly measured on the same current binary.
Later graph commits are outside this fixed snapshot. Hardware model and RAM were
not obtained: `sysctl` was denied. No hardware model is inferred from compiler flags.

The live guide and frozen guide snapshots identify the precise recipe measured.
Previous scripts/data and receipts remain in their original archive; this change
does not rewrite historical results. See [README.md](README.md) for commands and
[the original attribution](../2026-10-26-scifact-hybrid/README.md#dataset-and-model-attribution).

Next, profile the search adapter's independently constructed keyword SQL on its
bundled SQLite before applying this rewrite there. It still uses rowid `IN`;
its dense query and active graph work are separate from this measured guide.
Deployment-specific scope distributions/read-write mix, memory/cold-cache costs
and downstream agent outcomes remain unverified. Push and CI stay deferred.
