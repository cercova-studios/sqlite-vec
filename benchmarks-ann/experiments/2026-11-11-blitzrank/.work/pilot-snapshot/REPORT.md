# SciFact BlitzRank result

**Keep BlitzRank experimental.** Average nDCG@10 improved for BM25, dense and
RRF candidates, but the primary comparison against the strongest baseline did
not meet its confidence gate. Reranked p95 request latency was 18–21 seconds.
No default retrieval, adapter or C-core behavior changed.

Measured 2026-11-11 using OpenAI `gpt-6-luna`, reasoning effort `low`, and the
pinned upstream BlitzRank tournament algorithm. All **90/90 requests succeeded**,
with **322 model calls**, no retries, no malformed permutations and no model
fallbacks. Replaying every recorded comparison reproduced the actual schedules,
top-10 identities and call counts. Read [the method and commands](README.md) and
[frozen protocol/provenance](PROTOCOL.md).

## Relevance

The corpus contains all 5,183 SciFact documents. Each method admits its first
50 candidates and returns ten results. These 30 evaluation queries are a
previously selected subset of the earlier experiment's evaluation queries,
drawn from public training annotations. They are not a fresh blind benchmark.
The original 50 tuning queries selected **dense** as the strongest baseline.

| Pipeline | nDCG@10 | MRR@10 | Recall@10 | Labelled candidate recall |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.682865 | 0.654167 | 0.788889 | 0.833333 |
| BM25 + BlitzRank | 0.741263 | 0.708333 | 0.833333 | 0.833333 |
| Dense | 0.702716 | 0.689815 | 0.833333 | 0.933333 |
| Dense + BlitzRank | 0.779391 | 0.756944 | 0.866667 | 0.933333 |
| RRF | 0.677703 | 0.643651 | 0.800000 | 0.966667 |
| RRF + BlitzRank | 0.753908 | 0.721984 | 0.883333 | 0.966667 |

The primary comparison, **RRF + BlitzRank minus dense**, is **+0.051192**
nDCG@10, with paired query-bootstrap 95% interval **[−0.056799, +0.165461]**.
The predeclared gate required at least +0.02 and an interval above zero. The
mean clears the magnitude target; the interval crosses zero, so **the relevance
gate is not met**. This is an inconclusive pilot, not a verification failure.

| Descriptive same-candidate comparison | nDCG@10 difference | Paired 95% interval |
| --- | ---: | --- |
| BM25 + BlitzRank minus BM25 | +0.058398 | [+0.003562, +0.118040] |
| Dense + BlitzRank minus dense | +0.076675 | [−0.032804, +0.184815] |
| RRF + BlitzRank minus RRF | +0.076205 | [+0.012022, +0.146883] |

Intervals use 10,000 paired resamples, seed 20261111. Secondary comparisons are
descriptive and are not corrected for multiple comparisons. Dense + BlitzRank
has the highest observed mean; this pilot does not establish it as the best
pipeline for new queries.

The primary comparison improves eight queries, worsens six, and leaves sixteen
unchanged. Retained counterexamples include:

| Query ID | Dense nDCG@10 | RRF + BlitzRank nDCG@10 |
| --- | ---: | ---: |
| 876 | 1.000000 | 0.289065 |
| 434 | 0.430677 | 0.000000 |
| 391 | 1.000000 | 0.630930 |

The raw candidate identities, responses and judgments remain available for
these queries. Reranking never changes candidate recall or recovers an absent
document. There is no graph expansion or cross-encoder/late-interaction stage.

## Latency

| First stage | Baseline p50 / p95, ms | With BlitzRank p50 / p95, seconds |
| --- | ---: | ---: |
| BM25 | 2.615 / 5.487 | 13.931 / 18.483 |
| Dense | 8.452 / 9.850 | 10.045 / 19.339 |
| RRF | 11.076 / 14.535 | 13.087 / 21.468 |

Baseline timings contain 90 observations per method: 30 queries × three
repetitions. Reranked timings contain 30 observations per method, one live pass
per query. BGE query encoding is included where needed; document encoding is
offline. Reranked timing also includes retrieval, text fetch, token preparation,
HTTP round trips, hosted inference, graph work and durable response logging.
Initialization, input hashing and model warmup are outside request timers.

These are warm-host, sequential, end-to-end observations. They are not isolated
server inference latency, cold-disk measurements, repeated reranking trials or
concurrent-serving results. Different query subsets and API conditions can
change p95 substantially. The sum of timed reranked requests is 1,163.263 seconds;
that sum excludes preparation and pauses between the first request and the
remaining requests and is not a total wall-clock claim.

## Tokens and API cost

| Reranked first stage | Model calls | Input tokens | Output tokens | Reasoning subset | Estimated USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 110 | 782,593 | 32,060 | 22,674 | $0.0942893 |
| Dense | 104 | 745,907 | 25,563 | 16,741 | $0.0873722 |
| RRF | 108 | 777,602 | 31,731 | 22,510 | $0.0936257 |
| Total | 322 | 2,306,102 | 89,354 | 61,925 | **$0.2752872** |

Total usage is **2,395,456 tokens**. Reasoning is already included in output
tokens. The API reported no cached input tokens. Average calls per query are
3.667, 3.467 and 3.600 for BM25, dense and RRF respectively; average total tokens
are 27,155, 25,716 and 26,978.

The estimate uses the standard rates checked on 2026-11-11: $0.10 per million
input tokens and $0.50 per million output tokens. It is not a provider invoice.
The durable ledger charges a conservative $0.125 input allowance and $0.50 output
rate: **$0.33293975**, with **zero unsettled requests**, against the authorized
**$5 cap**. Every ledger entry maps to exactly one retained response. Baselines
make no paid model calls; local BGE compute and electricity are not priced.
[Official model/pricing documentation](https://developers.openai.com/api/docs/models/gpt-6-luna)

No sliding-window or pairwise LLM baseline was run, so these results do not
establish a BlitzRank token-saving percentage against other LLM schedulers.

## Preference cycles and input coverage

Top-10 output included a nontrivial strongly connected component in 29/30 BM25,
28/30 dense and 29/30 RRF requests. The median largest such component contained
6.5, 8 and 7 documents respectively; maxima were 22, 24 and 23. The algorithm
handles these as preference tiers. Their presence demonstrates cyclic observed
judgments; it does not prove those documents have equal ground-truth relevance.
Within tied tiers, the upstream algorithm's stable candidate order influences
the chosen order and boundary selection.

Only three distinct documents exceeded the 1,024-token document limit: three
candidate appearances in BM25, one in dense and two in RRF. These were explicitly
truncated before model calls. The experiment did not sweep window sizes,
truncation, reasoning effort, prompt wording or fusion weights after outcomes.

## Verification and remaining limits

Fresh checks in this task:

1. Eight unit/contract tests passed, including rejection before an over-budget
   API request, persistence across budget reload, malformed ranking refusal,
   upstream prompt parity and a known-order oracle for the actual sorter.
2. All 600 recomputed baseline query/method candidate lists matched the prior
   receipts before the 50-document shortlist. All 270 live baseline timing
   calls retained the same candidate identities.
3. Independent float64 NumPy checks covered 150 native L2 distances on three
   tuning queries. Maximum error was below `6e-8`; no strictly nearer candidate
   was absent from the returned set.
4. The final audit checked all 90 rankings, 322 complete model permutations,
   frozen prompt text/parameters, independent relevance arithmetic, source/input
   hashes and one-to-one budget entries. Replaying the upstream graph matched
   every observed ranking and schedule. No interrupted journals remain.

This covers one model alias, one window, one small previously examined training
subset and one reranking pass. Model/data contamination is unaudited. The full
150-query evaluation, repeated latency, model comparisons, total process/model
memory, production workloads and real agent answer quality remain unmeasured.
There is no evidence here to change adapter or C-core defaults.

Measurements are complete; archival review remains open. The new harness,
receipts and ignored inputs are retained as a protected review workspace because
they have no authorized Git recovery commit yet. Do not retire their unique
data as part of older-experiment cleanup. No commit, push or wiki publication
was performed by this experiment task.
