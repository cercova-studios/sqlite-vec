# Full 150-query SciFact BlitzRank result

**The full evaluation clears the relevance gate.** RRF + BlitzRank improves nDCG@10 by **+0.089385** over the tuning-selected dense baseline, with paired 95% interval **[+0.042886, +0.138554]**. This supports an optional host-side reranker experiment; it does not justify changing native or adapter defaults. Request p95 remains roughly 19–21 seconds.

Measured on 2026-11-11 with `gpt-6-luna`. All **450/450 query/method requests succeeded**, covering all **150 evaluation queries** and **1,622 model calls**. The set includes the original 90 pilot requests without repeat inference and 360 new requests. There were no retries, malformed permutations, failed requests or model fallbacks. [Protocol](PROTOCOL.md) and [reproduction guidance](README.md) describe the fixed settings.

## Relevance

| Pipeline | nDCG@10 | MRR@10 | Recall@10 | Labelled candidate recall |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.667205 | 0.630579 | 0.788333 | 0.844444 |
| BM25 + BlitzRank | 0.766705 | 0.743907 | 0.842222 | 0.844444 |
| Dense | 0.708804 | 0.675291 | 0.833333 | 0.915556 |
| Dense + BlitzRank | 0.794042 | 0.770167 | 0.875556 | 0.915556 |
| RRF | 0.693716 | 0.654026 | 0.818889 | 0.934444 |
| RRF + BlitzRank | 0.798189 | 0.769563 | 0.894444 | 0.934444 |

All 5,183 SciFact documents are present. Each first stage admits its first 50 candidates for reranking and returns ten. The 50 original tuning queries select dense as the strongest baseline; no selection is made from these 150 evaluation outcomes.

The primary comparison is **RRF + BlitzRank minus dense**: **+0.089385**, interval **[+0.042886, +0.138554]**. It meets both unchanged targets: at least +0.02 absolute nDCG@10 and a paired 95% lower bound above zero. Intervals use 10,000 paired query resamples, seed 20261111. They condition on the saved model decisions and do not measure inference-to-inference variance.

| Descriptive same-candidate comparison | nDCG@10 gain | Paired 95% interval |
| --- | ---: | --- |
| bm25 + BlitzRank minus bm25 | +0.099500 | [+0.063267, +0.137746] |
| dense + BlitzRank minus dense | +0.085238 | [+0.046221, +0.125705] |
| rrf + BlitzRank minus rrf | +0.104473 | [+0.066438, +0.145574] |

RRF + BlitzRank and dense + BlitzRank have similar observed means (0.798189 and 0.794042). This experiment does not establish that their small difference generalizes. Secondary comparisons are descriptive and are not corrected for multiple comparisons.

## Pilot and extension

The 30-query pilot had a primary gain of +0.051192 with interval [−0.056799, +0.165461], so it was inconclusive against dense. The human then requested expansion to all 150 queries. Model, prompts, candidate limits, scoring and gates remained fixed. The original pilot protocol, 90 exact receipts and 322 charge entries are retained with hashes; inherited full-run records identify their original receipt.

The **additional 120 queries alone** have a descriptive primary gain of **+0.098933**, interval **[+0.047843, +0.152014]**. The combined result and this subset are not fresh blind confirmatory tests: the extension followed observed pilot outcomes, the corpus uses public training annotations, and these queries were examined in earlier local retrieval experiments. Model contamination remains unaudited.

Pilot counterexamples remain present: query 876 fell from dense nDCG@10 1.0 to RRF + BlitzRank 0.289065; query 434 fell from 0.430677 to 0.0. A positive mean does not remove individual regressions.

## Latency

| First stage | Baseline p50 / p95, ms (150 queries) | Pilot p50 / p95, s (30 sequential queries) | Extension p50 / p95, s (120 queries, 3 workers) |
| --- | ---: | ---: | ---: |
| bm25 | 2.945 / 5.473 | 13.931 / 18.483 | 12.062 / 19.038 |
| dense | 8.263 / 9.786 | 10.045 / 19.339 | 12.141 / 19.487 |
| rrf | 11.430 / 14.684 | 13.087 / 21.468 | 11.702 / 19.147 |

Baseline timings contain 450 observations per method (150 queries × three repetitions). Pilot and extension each have one reranking pass per query; their latency strata are deliberately separate. This is not a controlled estimate of the effect of concurrency because the subsets and request times differ. No uniform sequential p95 over all 150 is claimed.

Timers include live BGE query encoding where required, exact SQLite retrieval, text fetch, token preparation, HTTP round trips, hosted inference, graph work and durable response logging. Session/model loading, input verification, warmup and executor queue wait are excluded. The three-worker extension gives each worker its own read-only SQLite connection and BGE session. No cold-cache, load-sweep, production-serving or repeated reranking-latency qualification was performed.

## Tokens and total spend

| Reranked first stage | Model calls | Input tokens | Output tokens | Reasoning subset | Estimated USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 542 | 3,862,369 | 147,195 | 101,052 | $0.4598344 |
| dense | 535 | 3,859,867 | 148,794 | 103,135 | $0.4603837 |
| rrf | 545 | 3,914,584 | 149,662 | 103,238 | $0.4662894 |
| Total including pilot | 1,622 | 11,636,820 | 445,651 | 307,425 | **$1.3865075** |

**Total tokens: 12,082,471.** Reasoning is already part of output tokens; the API reported zero cached input tokens. Standard-price estimates use $0.10/million input and $0.50/million output, checked against the [official model page](https://developers.openai.com/api/docs/models/gpt-6-luna) on 2026-11-11. This is an estimate, not a provider invoice.

**Conservative all-run accounting is $1.677428**, including the original pilot’s $0.33293975, against the original **$5 cap**. All 1,622 ledger entries are settled and map to exactly one received model response. The ledger was continued, not reset. Input was conservatively charged at $0.125/million to allow for cache-write pricing. The original 322 entries remain byte-for-byte as the prefix. No additional paid calls were made during verification.

Baselines make no paid model calls. Local embedding compute, electricity and document ingestion are not priced. No sliding-window or pairwise LLM baseline was run, so this does not measure BlitzRank token savings against other LLM schedulers.

## Cycles and truncation

| First stage | Requests with nontrivial top-10 tier | Median largest top tier | Maximum top tier | Truncated candidate appearances |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 141/150 | 8 | 28 | 14 |
| dense | 138/150 | 6 | 27 | 12 |
| rrf | 146/150 | 8 | 27 | 13 |

Cycles are observed model preferences, not proof of equal ground-truth relevance. The upstream sorter groups cycles into tiers; its stable candidate ordering affects ties and boundary selection. Passages over 1,024 `o200k_base` tokens are explicitly truncated. Candidate recall is unchanged by reranking, and missing first-stage documents cannot be recovered.

## Verification and interpretation

Fresh checks passed: ten warning-strict contract tests; all 600 baseline candidate-list comparisons against retained provenance; 1,350 live baseline identity checks; 150 native L2 distances against independent float64 NumPy calculations (maximum error below 6e-8); all 450 graph replays and 1,622 complete permutations; independent relevance arithmetic; fixed input/prompt/source identities; inherited pilot equality; and exact continuing budget accounting. The replay removes the API key and denies Python connection/DNS calls.

The relevance gate passes for this fixed 150-query configuration. It is evidence for an optional host-side reranking stage, with a substantial seconds-per-query latency tradeoff. It does not qualify production rollout, native integration, agent answer quality, generalization across corpora/models, repeated-run stability or resource behavior on constrained devices. Model alias drift and training-data contamination remain possible.

No adapter, C core, ranking defaults, main branch or CI configuration changed. Results and harness recovery are prepared on isolated ancestry from 812a5a3 for independent review; publication is owned by the integration chat. The recovery instructions describe an offline replay from committed objects, distinct from repeating paid inference.
