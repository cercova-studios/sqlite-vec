# Public SciFact hybrid retrieval evaluation

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

Completed 2026-10-27; experiment directory retains its 2026-10-26 start date.

## Decision

Keep current defaults. On this public labelled corpus with real BGE embeddings, hybrid and vector-only have nearly identical full-corpus nDCG@10. The hybrid-minus-vector paired query bootstrap interval includes zero in both contexts. Hybrid improves over the SQLite keyword baseline here, but does not establish an advantage over dense retrieval. No engine, rank-weight, candidate-policy, ANN or compression change follows.

## Full results

All 300 BEIR test queries are evaluated against all 5,183 abstracts and 339 positive judgments. The scoped overlay retains 3,091 eligible documents and leaves 182 queries with at least one eligible positive judgment. Recall, nDCG and MRR are macro averages over queries with remaining positive judgments. Do not interpret higher scoped scores as improvements over the full corpus: the candidate set and averaging cohort both change.

| Context | Method | Scored queries | Recall@10 | nDCG@10 | MRR@10 | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| full | vector | 300 | 0.8362 | 0.7127 | 0.6822 | 2.36 | 2.61 |
| full | keyword | 300 | 0.7989 | 0.6695 | 0.6354 | 1298.57 | 2623.85 |
| full | hybrid | 300 | 0.8387 | 0.7130 | 0.6800 | 1278.45 | 2625.02 |
| scoped | vector | 182 | 0.8846 | 0.7649 | 0.7296 | 2.20 | 2.35 |
| scoped | keyword | 182 | 0.8214 | 0.7094 | 0.6789 | 787.61 | 1562.78 |
| scoped | hybrid | 182 | 0.8723 | 0.7586 | 0.7246 | 789.38 | 1587.57 |

Latency uses the same fixed sample of 30 queries in each context, chosen with seed 42 without inspecting relevance outcomes. Each method has one warmup and five timed rounds per sampled query, with method order shuffled. The p50 and nearest-rank p95 pool 150 observations per method/context, including queries without eligible positive judgments. These are descriptive warm SQL timings on one development host, not production tail estimates or confidence intervals. Embedding generation and query tokenization are excluded.

Full-corpus keyword and hybrid median latencies are about 1.3 seconds, versus 2.4 ms for vector-only. This is the published scoped SQL recipe, including the rowid eligibility subquery, not a claim that unconstrained FTS5 has the same cost. The raw receipts retain EXPLAIN QUERY PLAN for the first query in each context and per-call CPU/wall timings. No CPU/off-CPU profile establishes the root cause. Investigate this filtered keyword query path before changing ranking or C engine code.

## Paired relevance differences

Intervals resample paired per-query nDCG differences 10,000 times (seed 42). They describe query resampling for this fixed dataset/model and do not account for model selection, shared evidence documents or application shift. No test labels were used to train or tune the model, fusion weights or candidate budget in this experiment.

| Context | Hybrid minus | Mean nDCG delta | Query bootstrap 95% interval | Wins / ties / losses |
|---|---|---:|---|---|
| full | vector | +0.0003 | [-0.0250, +0.0256] | 54 / 200 / 46 |
| full | keyword | +0.0436 | [+0.0211, +0.0659] | 73 / 199 / 28 |
| scoped | vector | -0.0063 | [-0.0423, +0.0272] | 30 / 125 / 27 |
| scoped | keyword | +0.0492 | [+0.0177, +0.0817] | 46 / 121 / 15 |

## Data, embeddings and query formation

- Public BEIR SciFact, distributed at the [official dataset URL](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip). Its MD5 matches the [BEIR published checksum](https://github.com/beir-cellar/beir/wiki/Datasets-available); SHA256 is `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`. The entire corpus and all 300 labelled test queries are used. Train judgments are not used. There is no query/document ID overlap. All 339 relevance grades are 1.
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), pinned to `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The official ONNX model runs locally with CPUExecutionProvider, four intra-op threads, one inter-op thread and batches of 16. CLS pooling and L2 normalization follow the model documentation. Queries receive the documented retrieval instruction; documents use title, one space and abstract without an instruction.
- Embeddings have 384 float32 dimensions and a 512-token cap. All 5,483 vectors passed finite/unit-norm checks. A two-document batch-versus-separate padding check passed. 459 documents reached the token cap; no query did. Preparation took 262.15 seconds, excluding downloads, and is outside retrieval timing. The tokenizer cache was independently compared byte-for-byte with the pinned public tokenizer.
- Exact flat L2 search preserves the engine default metric. Normalized vectors make L2 ranking equivalent to cosine in exact arithmetic; the actual float32 L2 outputs are checked. No ANN index, quantizer or reranker is used.
- FTS5 uses its default tokenizer and BM25 rank column over full title plus abstract. Query terms come from SQLite unicode61 tokenization, deduplicated in original order and individually quoted with OR between terms. Operators in claim text are literal tokens. No stemming, custom stopword list or rank tuning is added. Unlike dense input, keyword input is not truncated at 512 tokens. These choices are recorded and need not match another BEIR implementation.
- The existing guide supplies the actual scoped RRF SQL. Both branches filter eligibility before their fixed budget of 50; output is ten. RRF keeps equal weights and constant 60. Vector-only and keyword-only execute their full 50-candidate budgets before Python takes ten; an early-LIMIT standalone query could cost less. This is not a new global default budget.

## Scope, lifecycle and correctness

The public corpus has no tenant/validity fields. The overlay is explicitly artificial: the first SHA256 byte of the document ID modulo ten assigns wrong tenant, expired at 100, future from 101, invalidated, or eligible. It does not consult relevance labels or embeddings. At as_of=100 there are 543 wrong-tenant, 527 expired, 517 future, 505 invalidated and 3,091 eligible documents. Texts, vectors, global keyword statistics and original relevance labels remain unchanged. Scoring intersects positive judgments with eligibility rather than inventing new semantic labels.

All 1,800 query/method/context points passed eligibility and provenance assertions. There were zero returned ineligible IDs. Exhaustive float64 L2 over the stored float32 inputs checks vector candidates, with 1e-6 cutoff-tie tolerance and 2e-5 reported-distance tolerances. Separate branch queries and Python RRF check hybrid IDs, ranks, scores, distances and source IDs. trec_eval independently matched all 4,338 answerable metric values for recall, nDCG and reciprocal rank.

All methods returned results for every query in both contexts. In the overlay, 118 queries have no remaining positive judgment but still receive results. Sparse positive-only judgments mean unjudged results are treated as zero gain for scoring, not proven irrelevant. Nonempty retrieval is not an abstention or usefulness guarantee. Eligibility predicates are not an authorization boundary.

Five tests passed: known and graded metric cases, scope boundaries, literal FTS operators, and pending invalidation/savepoint rollback/full rollback/commit/reopen over real embeddings. The measured run reopens after each context and checks committed eligibility. It does not retime every query after reopen. Full application suites and sanitizers were not rerun for this benchmark-only change.

## Write and storage probes

Initial insertion and commit of all three stores took 143.96 ms. Full eligibility assignment took 4.13 ms; applying the overlay took 6.45 ms. Each assignment issues 5,183 scope UPDATEs in one transaction. These are single-operation probes, not sustained write throughput.

Allocated bytes were 24,244,224 after full assignment and 24,260,608 after the overlay. FTS optimize, measured after retrieval, took 25.90 ms; afterward the file retained 28,397,568 allocated bytes with 1,067 free pages. No documents were deleted, no VACUUM ran, and these are SQLite file allocations, not Lance version-retention measurements.

## Provenance, limits and next work

The [README](README.md) contains commands, attribution and artifact roles. Saved receipts include query IDs, public qrels, lexical expressions, complete branch candidates, final rows, source IDs, timings, plans and storage costs. Data/model/array hashes are in the [data manifest](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-10-26-scifact-hybrid/recorded/data-manifest.json); C source/header hashes, excluded dirty C diff, compiler and flags are in the [build manifest](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-10-26-scifact-hybrid/recorded/build-manifest.json). The [guide snapshot](recorded/guide.md) preserves the measured recipe. Downloaded datasets, weights, arrays and compiled libraries remain ignored in `.work/` for replay; none are committed.

The C snapshot is `fee2050362831a4d65fec74c2a82b38a74a888d8`, excluding the unrelated HNSW helper edit. Compilation targets Apple M1/NEON at O3 and retains the existing DiskANN unused-parameter warning. Runtime: macOS 27.0.1 arm64, Python 3.12.13, NumPy 2.1.3, SQLite 3.53.1, ONNX Runtime 1.23.2 and tokenizers 0.22.1. The metric oracle uses pytrec-eval-terrier 0.5.11.

This is one historical public scientific retrieval corpus and one pretrained model. It is not production workload validation, original blind-test reproduction, claim-veracity classification or downstream agent evaluation. Model training/test overlap was not independently audited. Token truncation, tokenizer/query formation and sparse judgments limit comparisons with published scores. No knobs were selected from these results.

Timings are warm, sequential, single-client and disk-backed with SQLite defaults and DELETE journaling. CPU affinity/frequency, actual CPU model and other development work were uncontrolled; no dedicated-machine, cold-cache, concurrent-client, RSS, I/O or cross-platform measurements were collected. The timing sample is smaller than the relevance set. Do not extrapolate latency or ranking gains to other workloads.

Next: profile the measured filtered FTS query path and verify any proposed SQL-plan change against these fixed labels, candidates and eligibility checks. Keep ranking defaults unchanged. A deployment decision still needs its own corpus/read-write mix and application outcomes. Shared source notes were inaccessible; primary dataset/model sources and current code were inspected directly.
