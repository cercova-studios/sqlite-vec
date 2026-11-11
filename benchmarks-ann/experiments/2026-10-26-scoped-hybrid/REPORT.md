# Scoped hybrid retrieval evaluation — 2026-10-26

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

## Decision

Keep the existing recipe and defaults. This controlled evaluation validates scoped eligibility and lifecycle behavior, but demonstrates no universal relevance benefit from hybrid fusion or larger candidate budgets. Production-corpus evaluation remains the next decision gate. No engine change, alpha tuning, push or CI is included.

At baseline with no distractors, hybrid macro recall@10 is 96.4%, versus 82.1% for vector-only and 61.9% for FTS-only. At 10,000 distractors, hybrid and vector-only both reach 53.6%, while FTS-only reaches 0% on these deliberately adversarial keyword matches. Those percentages describe authored coordinates and labels, not the quality of a real embedding model.

## Workload and reference

- Sixteen hand-authored procedure records cover deployment, credential rotation, refunds and backups. Each topic includes a close paraphrase without the query keyword, keyword-bearing close records and a relevant keyword-bearing record far from the query vector. [Fixture](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-10-26-scoped-hybrid/fixture.json) labels relevance independently of retrieved ranks. All four-dimensional vectors are controlled coordinates, not learned embeddings.
- Nine queries include four topic searches, valid-from/valid-to boundaries, a missing keyword match, an absent tenant and an unanswerable question within an eligible tenant. Macro recall excludes the two queries without relevant eligible records; each of the remaining seven query forms has equal weight. Relevance sets change when records become ineligible or are replaced.
- Corpus sizes are 16, 1,016 and 10,016 records. Added records are irrelevant; at time 100, 40% belong to another tenant, 20% have expired, 20% start in the future, 10% are invalidated and 10% are eligible distractors. Earlier/later boundary queries change those eligibility proportions. Ineligible distractors have exact query vectors; eligible distractors have seeded random vectors and keyword-bearing unrelated text.
- Identical documents are inserted in three orders (seeds 0/1/2). Branch budgets are 10/50, output limit 10. Flat float32 vec0 search avoids ANN approximation as a confound. RRF uses the published equal weighting and constant 60. Larger candidate budgets can promote irrelevant records present in both branches.
- Eligibility is checked by a separate Python predicate against fixture/generated attributes. Relevant IDs are selected from authored labels and that predicate, never from rankings. Vector candidates are checked by exhaustive Python distances over float32 inputs, including ties. Hybrid IDs, ranks, distance nullness, scores and source IDs are checked against separate branches and Python RRF. FTS5 itself supplies keyword ranking; this is not an independent implementation of BM25.
- Invalidation sets a non-NULL timestamp, which the recipe excludes at every as_of value; it does not reconstruct historical pre-invalidation visibility. The replacement updates scope provenance, text and vector in one transaction. These are retrieval predicates, not an authorization boundary.

## Recall and latency

Baseline values pool the three insertion orders. Recall is the arithmetic mean over 21 answerable query/order pairs. Latencies pool nine timed executions for all nine queries per order, including empty results, after two warmups per method/query. Method order is shuffled per round. The median and nearest-rank p95 describe this mixed workload; nine repeats per query do not estimate a production tail or confidence interval. Vector-only and FTS-only materialize the full branch budget before Python takes ten; an optimized early-LIMIT single-branch query could cost less.

| Distractors | Branch budget | Method | Macro recall@10 % | p50 ms | p95 ms |
|---:|---:|---|---:|---:|---:|
| 0 | 10 | vector | 82.1 | 0.016 | 0.020 |
| 0 | 10 | keyword | 61.9 | 0.064 | 0.077 |
| 0 | 10 | hybrid | 96.4 | 0.110 | 0.132 |
| 0 | 50 | vector | 82.1 | 0.017 | 0.022 |
| 0 | 50 | keyword | 61.9 | 0.065 | 0.081 |
| 0 | 50 | hybrid | 96.4 | 0.114 | 0.136 |
| 1,000 | 10 | vector | 53.6 | 0.066 | 0.097 |
| 1,000 | 10 | keyword | 0.0 | 0.507 | 1.765 |
| 1,000 | 10 | hybrid | 53.6 | 0.605 | 1.978 |
| 1,000 | 50 | vector | 53.6 | 0.074 | 0.105 |
| 1,000 | 50 | keyword | 0.0 | 0.515 | 1.790 |
| 1,000 | 50 | hybrid | 39.3 | 0.639 | 2.041 |
| 10,000 | 10 | vector | 53.6 | 0.658 | 1.143 |
| 10,000 | 10 | keyword | 0.0 | 16.411 | 102.187 |
| 10,000 | 10 | hybrid | 53.6 | 17.038 | 104.528 |
| 10,000 | 50 | vector | 53.6 | 0.685 | 1.208 |
| 10,000 | 50 | keyword | 0.0 | 16.436 | 102.491 |
| 10,000 | 50 | hybrid | 53.6 | 17.146 | 104.672 |

With 1,000 distractors, increasing hybrid branch budget from 10 to 50 reduces baseline recall from 53.6% to 39.3%. Candidate overlap can reward irrelevant keyword/vector matches; retrieving more candidates is not a monotonic relevance improvement. This is a counterexample, not evidence to change the default budget.

At 10,000 distractors the keyword path has a 16.4 ms median, compared with 0.66 ms for vector-only; hybrid is about 17.0 ms at budget 10. Boundary queries with more eligible distractors contribute to the pooled p95. [Query plans](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-10-26-scoped-hybrid/recorded/query-plans.json) preserve the observed plans for the main and before-start queries. No CPU/off-CPU profile was collected, so these timings do not establish the cause of scaling. Investigate the filtered FTS query path before considering engine tuning.

## Eligibility, empty results and lifecycle

The complete run covers 1,944 quality points and 17,496 timed queries: three corpus sizes × three insertion orders × four states × nine queries × two budgets × three methods × nine timing rounds. All states returned zero ineligible or stale (expired/invalidated) records. Vector distance and fusion/provenance assertions passed at every query/budget/state/order.

The four measured states are baseline, committed invalidation, committed replacement, and reopened connection. After reopen, a separate check confirms identical result IDs to the updated state. Six evaluation tests additionally check explicit boundary judgments, read-your-writes, savepoint and full rollback, committed storage contents across scope/text/vector stores, reopened answers, fixed corpus values across seeds, and deliberately unsafe filter mutations. Removing tenant, invalidation, valid-from or exclusive valid-to filtering causes the eligibility oracle to reject results.

The deliberately late-filtered global-candidate control returns zero candidates at 1,000 and 10,000 distractors for the deployment query, across all three orders and all four states. The safe query still returns relevant results. This confirms candidate starvation even when the final filter prevents ineligible output. At zero distractors, the late control has no such starvation; its union is diagnostic and is not a ranked top-10 comparator.

Across all measured states/budgets/sizes, vector and hybrid empty-result rates are 1/9, while FTS-only is 3/9. For answerable queries, vector/hybrid empty-result rates are zero and FTS-only is 1/7 (the paraphrase query has no keyword matches). The unanswerable question still receives unrelated vector/hybrid results in every configuration. Returning results is not evidence of relevance or an abstention guarantee.

## Write, maintenance and retained storage costs

Values below are medians over three insertion orders. Each cost is one operation per database, not a sustained write-throughput benchmark. Insert cost writes all three stores and commits. Invalidation updates scope only; replacement writes scope, vector and text, then commits. FTS optimize is measured after query timings. Files use SQLite DELETE journaling; allocated database bytes are retained file space, not live payload or Lance historical-version bytes. No VACUUM or version deletion is performed.

| Distractors | Insert + commit ms | Invalidate + commit ms | Replace + commit ms | FTS optimize ms | Final file bytes |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.559 | 0.340 | 0.472 | 0.672 | 81,920 |
| 1,000 | 6.338 | 0.672 | 0.922 | 0.835 | 184,320 |
| 10,000 | 67.058 | 0.699 | 0.995 | 1.313 | 1,228,800 |

All original records remain stored after invalidation. Raw receipts preserve page counts/free pages before and after each phase and optimize; file sizes can vary by insertion order. The short run does not measure long-term retention growth, WAL traffic, Lance optimize or storage reclamation.

## Provenance and verification

Run commands and artifact roles are in [README](README.md). Frozen [raw receipts](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-10-26-scoped-hybrid/recorded/receipts.json.gz), [all 72 summaries](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-10-26-scoped-hybrid/recorded/summary.json), [build manifest](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-10-26-scoped-hybrid/recorded/build-manifest.json), [published guide snapshot](recorded/guide.md) and query plans are retained. `manifest.json` hashes the archive. `verify.py` checks those hashes and regenerates eligibility, relevance and summary metrics from raw IDs and timings.

The extension was built from committed C sources at `52d1f144a75b2ce8faeb3c2ce21cc42b15b99c12` in a temporary directory. The existing HNSW helper edit was excluded and is recorded in the build manifest. Compilation used Apple M1/NEON at O3; the existing DiskANN unused-parameter warning remains. Runtime: macOS 27.0.1 arm64, Python 3.12.13, SQLite 3.53.1. Source/library/harness/fixture/guide hashes are recorded. No production source was edited.

Fresh checks: six lifecycle/oracle tests and three existing vec_rrf loadable tests passed. The latter used the working-tree default loadable build; the six new tests and measured sweep used the committed-source library identified above. Full application suites, sanitizers and other builds were not rerun for this benchmark-only change.

## Limits and next step

This is a controlled synthetic benchmark, not production relevance validation. Sixteen relevance-labelled documents, four authored dimensions, single-term FTS queries, one skew pattern and short local runs cannot represent realistic document retrieval, learned embeddings, large tenants or downstream agent outcomes. Recall drops at earlier/later times partly reflect deliberately eligible but irrelevant exact-vector distractors. Equal weighting of the query forms is a workload choice.

Timings are warm, sequential and single-client on local disk, including Python sqlite3/fetch overhead; reopening is warmed before measurement. CPU affinity/frequency and other applications are uncontrolled. CPU samples were retained alongside wall timings, but no system saturation, memory, I/O, concurrency, cold-cache, RSS or cross-platform profile was collected. No performance optimization or release claim follows.

The next useful step is a representative labelled corpus with real embeddings and a specified read/write mix, retaining this eligibility/lifecycle oracle and the three retrieval baselines. Include selective tenants and realistic validity histories. Measure the filtered FTS query plans and per-query latency before choosing a ranker, branch budget or schema optimization. Downstream agent usefulness needs a separate real task workload. Shared source notes were inaccessible in the sandbox; local semantic skill discovery also failed to initialize Metal, so the supplied skills and current repository evidence were used.
