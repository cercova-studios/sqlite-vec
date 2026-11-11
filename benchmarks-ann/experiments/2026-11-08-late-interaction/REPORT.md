# SciFact late-interaction result

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

**Decision: stop before native MaxSim.** The frozen ColBERTv2 configuration
reduced relevance against the tuning-selected dense baseline. Faster numerical
scoring would not repair that ranking loss. Keep the reference scorer and receipts
for future experiments; do not add a C function or change default retrieval.

## Relevance

All 5,183 SciFact documents were indexed. Before ranking, 50 tuning queries and
150 evaluation queries were frozen from the 809 labelled **training** queries.
The previously used 300 test-query IDs were excluded. This is a local evaluation,
not a new blind benchmark. No model/window/weight changes followed the outcomes.

Dense retrieval won tuning nDCG@10: 0.815551, versus RRF 0.798809 and BM25
0.695526. The primary comparison was fixed in advance: RRF candidates reranked
with ColBERT versus that tuning-selected baseline.

| Evaluation pipeline | nDCG@10 | MRR@10 | Recall@10 | Labelled candidate recall |
|---|---:|---:|---:|---:|
| BM25 | 0.667205 | 0.630579 | 0.788333 | 0.877778 |
| BM25 + late | 0.622978 | 0.594619 | 0.732222 | 0.877778 |
| Dense | 0.708804 | 0.675291 | 0.833333 | 0.947778 |
| Dense + late | 0.643448 | 0.616852 | 0.754444 | 0.947778 |
| RRF | 0.693716 | 0.654026 | 0.818889 | 0.954444 |
| RRF + late | 0.634724 | 0.605323 | 0.758889 | 0.954444 |

The primary nDCG@10 difference is **−0.074080**, with paired query-bootstrap
95% interval **[−0.126369, −0.022455]** (10,000 samples, seed 20261110).
The required improvement was +0.02 with the interval above zero; both quality
conditions failed. Descriptive same-candidate differences are −0.044227 for
BM25, −0.065356 for dense, and −0.058992 for RRF. Their intervals and every
query's raw rankings are retained in `recorded/`.

Each baseline/late pair uses exactly the same candidates. BM25 and dense each
admit 100; RRF merges the top 50 from each branch with reciprocal-rank constant
60. Across all 200 queries, the union contains 68–97 documents (mean 85.335).
All admitted candidates are scored. Equal computed scores use ascending document
ID; scores are never rounded to create ties.

## Latency and memory

Measurements ran on macOS 27.2 arm64, Python 3.12.13, CPU only. Torch and ONNX
use four intra-operation threads; BLAS uses one. SQLite 3.53.1 uses a 2 MiB pager
cache and no mmap. The trusted prebuilt native bundle is from `949073f`; its hash
is pinned in `recorded/baseline-manifest.json`. Unrelated working-tree C edits
were not compiled into this experiment.

Each latency distribution contains 90 observations: 30 evaluation queries ×
three repetitions. Query and method order is shuffled with fixed seeds. Live
runs alternate baseline/late order by repetition. Models remain loaded; three
queries warm the encoders. OS filesystem caches were not flushed.

| Pipeline | Live total p50 / p95, ms | Precomputed-query rerank p50 / p95, ms |
|---|---:|---:|
| BM25 | 2.692 / 5.536 | — |
| BM25 + late | 140.229 / 143.834 | 125.757 / 130.898 |
| Dense | 8.377 / 9.648 | — |
| Dense + late | 145.985 / 148.360 | 126.129 / 131.300 |
| RRF | 11.228 / 14.323 | — |
| RRF + late | 130.719 / 140.564 | 107.642 / 118.336 |

Live totals include BGE query encoding when needed, ColBERT query encoding for
late interaction, BEGIN/ROLLBACK, candidate SQL, artifact/source validation,
digest checks, BLOB reads, decoding and scoring. Lexical query tokenization is
prepared outside these request timers for both variants. Live RRF-late p95
components are BGE 3.532 ms, ColBERT 12.275 ms, retrieval 10.956 ms and reranking
118.448 ms. Component percentiles need not sum to the total percentile.
Reranking accounts for a mean 82.5% of live RRF-late time. This identifies
material stage cost, not a prediction of native-kernel speedup: that timer also
includes validation, hashing and SQLite I/O.

Fresh scoring-process peak RSS is **102,989,824 bytes (98.219 MiB)**. Fresh live
process peak is **1,559,478,272 bytes (1,487.234 MiB; 1.452 GiB)** with both model
runtimes resident. The live process also encodes an eight-document maximum-input
batch to probe preparation memory. This is not the original full-ingestion peak,
which was not captured. Baseline-only model RSS was not isolated: paired live
baseline calls ran with both encoders resident. RSS is the macOS high-water value
from one fresh child, including imports and output serialization, not Python-only
allocation accounting. No concurrent serving or memory-pressure test was run.

The first RRF query on a new connection took 5.976 ms retrieval plus 113.899 ms
reranking. This is a first-process diagnostic with existing filesystem caches,
not cold-disk latency. Model loading took 2.884 seconds, outside warm query timers.

## Storage and preparation

| Quantity | Observed bytes | MiB |
|---|---:|---:|
| Baseline-only SQLite: sources + FTS5 + dense | 33,976,320 | 32.402 |
| Combined SQLite including token artifacts/references | 461,602,816 | 440.219 |
| Token matrix payload: 5,183 documents + 200 queries | 420,880,384 | 401.383 |
| ColBERT safetensors weights, outside SQLite | 438,349,816 | 418.043 |

The combined database is **13.586×** the baseline size. This ratio includes the
200 retained query matrices and metadata; it is not a per-document compression
ratio. Model weights, runtime installation, duplicate baseline comparison file
and experiment receipts are excluded from the serving-store gate. No model
weights, database, dataset or virtual environment are committed.
The combined fixture also exceeds search-lab's existing 64 MiB database admission.
This independent archive's 1 GiB research gate does not qualify CLI/MCP integration
or change that adapter limit.

ColBERT document encoding took 140.100 seconds; timed document-artifact insertion
transactions took 1.459 seconds. Full preparation took 150.352 seconds, including
query encoding and qualification. Baseline-only SQLite construction from existing
dense vectors took 0.306 seconds; this excludes their earlier model inference.
These are different preparation scopes, not an end-to-end ingestion speed ratio.
The measured matrices read per request reach 8,299,008 bytes including the query;
charged work reaches 69,266,208 units, below the frozen aggregate caps. These are
SQLite-returned payload/logical work counts, not physical filesystem read bytes.

Both preparation and baseline append completed WAL checkpoints; final persistent
WAL content is zero. Transient ingestion WAL high-water and update/history churn
were not measured. There is no claim about managed-writer publication, undo,
artifact garbage collection or storage under an update workload.

## Qualification and limits

`oracle.py` is a stdlib scalar reference with little-endian float32 inputs,
ordered float64 accumulation, negative maxima, full normalization checks and
budget rejection before decoding. A0 supplied it with independent relational and
hand-computed tests. `host_scorer.py` is the measured NumPy backend: ordered
component cumulative sums match the scalar exactly for all qualified cases,
including six full real query/document matrix pairs. It processes one document
at a time, with a document-sized float64 buffer and scratch. It avoids a query ×
document similarity matrix but does **not** satisfy the proposed native kernel's
constant/O(query-token) extra-scratch target. The scalar reference uses constant
extra scratch. Neither backend is a registered SQLite SQL function.

The host checks profile, role, source epoch/input digest, artifact digest and
aggregate work/bytes in one read transaction. Synthetic tests reject missing,
stale, duplicate and corrupted candidates without partial publication. Twenty-three
tests passed, including deliberately wrong maxima/omissions/order/budget policies
and receipt mutations. The independent analyzer recomputes all 600 quality rows'
metrics/work, checks all 810 timing observations and verifies sampled live
rankings against precomputed-query rankings. Hashes bind inputs and scripts;
they are provenance checks, not signatures or independent proof of model origin.

A0's independent review found query payload hashing/decoding before full request
admission. The host now checks candidate IDs/count and query metadata first,
admits aggregate work/bytes, then fetches the query BLOB. Regressions failed on
the earlier implementation and passed after the repair. Empty candidate sets
still validate/admit the query and return an empty result. Final score/live
measurements were rerun against the repaired source; preparation, model, split
and acceptance thresholds were unchanged. Initial/interim attempts remain in
`.work/` with their original script bytes; the correction receipt records their
identities and ranking parity with the final run in
[the correction receipt](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-late-interaction/recorded/admission-correction.json).

The encoder adapts the pinned MIT-licensed ColBERT implementation using standard
Transformers BERT and safe tensor weights. Token markers, attention/masking,
punctuation removal, projection and normalization were checked against inspected
upstream source. Expected token lists and batch-versus-single embedding checks
passed. The official ColBERT package's CPU path compiles a native MaxSim extension,
so it was not installed/imported. End-to-end numerical equivalence against that
package is unverified; this is a source-qualified adapter, not a certified upstream
benchmark reproduction. No downloaded Python source or remote model code ran.

The checkpoint's 180-token document limit truncates **4,730/5,183 documents
(91.3%)**; 32 query slots truncate **28/200 queries**. BGE uses a 512-token input
limit. This is a comparison of the frozen usable pipelines, not an experiment
isolating only the interaction mechanism. Truncation and domain/model suitability
are plausible explanations for the loss, not measured causes. Public training
data/model contamination was not independently audited. BGE document embeddings
were reused from ONNX Runtime 1.23.2; current query encoding uses 1.31.1 for both
baseline and reranked candidates. One checkpoint, split and machine cannot
establish that late interaction is generally ineffective.

## Gate and next decision

| Frozen gate | Result |
|---|---|
| Primary nDCG@10 gain ≥ +0.02; paired interval above zero | Fail / fail |
| RRF reranking p95 ≤ 250 ms; live total p95 ≤ 500 ms | Pass / pass |
| Scoring RSS ≤ 256 MiB; encoder RSS ≤ 2 GiB | Pass / pass |
| Combined SQLite ≤ 1 GiB | Pass |

Retain this negative result and stop L2–L4. A future investigation would need a
separate, preregistered quality hypothesis and fresh evaluation evidence before
native work. Larger inputs, new checkpoints, score blending, quantization and
candidate windows were not tuned after this result.

## Sources and receipts

- [Frozen protocol](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-late-interaction/recorded/protocol.json), [analysis plan](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-late-interaction/recorded/analysis-plan.json),
  [summary](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-late-interaction/recorded/summary.json), [retained hashes](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-late-interaction/recorded/manifest.json).
- [ColBERT model revision](https://huggingface.co/colbert-ir/colbertv2.0/tree/c1e84128e85ef755c096a95bdb06b47793b13acf).
- [Inspected ColBERT source revision](https://github.com/stanford-futuredata/ColBERT/tree/cc4f3dc91c0b45d2d08c251d9d95178285c65f1c),
  with source-specific URLs/hashes in [model provenance](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-late-interaction/recorded/model-manifest.json)
  and the retained [MIT license](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-late-interaction/UPSTREAM-LICENSE).
- [BEIR SciFact archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip),
  SHA256 `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.
- [Motivating OpenSearch article](https://opensearch.org/blog/boost-search-relevance-with-late-interaction-models/).
