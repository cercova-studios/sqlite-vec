# Full SciFact BlitzRank protocol

The 150-query extension was frozen before any additional inference at `2026-11-11T20:18:08Z`. Model/scoring settings and the primary gate are unchanged from the observed pilot. Expanding after the pilot was a human-requested continuation, not a fresh blind confirmatory test.

## Scope and accounting

- All 150 evaluation queries from the prior SciFact training-label split; 50 original tuning queries select the baseline.
- Three first stages per query: BM25, exact dense L2 and RRF. Return ten documents from the first 50 admitted candidates.
- Preserve the original 90 pilot requests, 322 model calls and $0.33293975 conservative ledger charge. Perform only the remaining 360 requests.
- One original $5 cap and one continuing ledger. Reservations and settlements are serialized by a lock; a process lock excludes overlapping run commands.
- Full protocol SHA256: `7ee59ef403dd837d77e86ed53ce5a906e332ca8a55de3dd9f86b1b66a4fea867`.
- Pilot protocol SHA256: `63febb9b5c5eb10cf9d51f13382c145b637b6355d0874739537c193c51aaa464`.
- Pilot ledger SHA256 before extension: `5286cd60576edb9c57f0a01a67e4e52ead49d9b90683d54f7e113b35b0fa337f`.
- Original pilot files remain byte-for-byte; derived reused rows add an origin path/SHA and bind to the extension protocol. Final verification removes that binding and requires exact equality with the original row.

## Fixed model and retrieval settings

| Setting | Value |
| --- | --- |
| `model` | `gpt-6-luna` |
| `endpoint` | `https://api.openai.com/v1` |
| `upstream_revision` | `076018cf6153608203029e29562c36f50273b348` |
| `selected_baseline` | `dense` |
| `corpus_documents` | `5183` |
| `candidate_limit` | `50` |
| `rrf_branch_candidates` | `50` |
| `rrf_constant` | `60` |
| `topk` | `10` |
| `window_size` | `20` |
| `max_document_tokens` | `1024` |
| `max_query_tokens` | `1024` |
| `tokenizer` | `o200k_base` |
| `max_rounds` | `50` |
| `max_output_tokens` | `2048` |
| `call_timeout_seconds` | `120` |
| `max_request_seconds` | `1800` |
| `max_run_tokens` | `20000000` |
| `temperature` | `omitted` |
| `reasoning_effort` | `low` |
| `retries` | `0` |
| `api_cap_usd` | `5` |
| `request_concurrency` | `3` |

Dense uses `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, 384 normalized float32 dimensions and scalar exact L2. The query prefix is `Represent this sentence for searching relevant passages: `, including its trailing space. BM25 uses SQLite FTS5 over title + one space + abstract, with quoted query terms joined by OR. RRF merges 50 per branch, constant 60, then admits its first 50 fused IDs. BM25/dense first retrieve 100, then admit their first 50; baseline top-ten order is unchanged.

The RankGPT prompt is identical to the pinned upstream prompt. Inputs use `o200k_base` prefix truncation at 1,024 tokens per document/query. Responses must be complete bracketed permutations; malformed, duplicate, missing and out-of-range IDs reject the request rather than being repaired. There are no automatic retries, model substitutions or baseline fallbacks.

Requests use Chat Completions, `service_tier:default`, `store:false`, no tools and no temperature override. The model is an alias; no immutable dated snapshot is listed. The 2,048-token completion limit includes reasoning tokens.

## Quality gate and timing

Primary comparison: RRF + BlitzRank minus the tuning-selected dense baseline in nDCG@10. Require at least +0.02 and a paired query-bootstrap 95% lower bound above zero; 10,000 resamples, seed 20261111. Secondary comparisons and the added-120 subset are descriptive. Model/data contamination is unaudited.

Baseline timings: 150 queries × 3 methods × 3 shuffled repetitions = 1,350 observations, with BGE query encoding included where applicable. Reranking retains the pilot's 90 sequential timings and records 360 new timings under three workers. Report those latency strata separately; do not interpret the combined distribution as a uniform sequential-service measurement. This is a fixed-concurrency batch, not a load sweep or concurrent-serving qualification.

Each worker owns its read-only SQLite connection and BGE session. Input verification, session loading and warmup are outside request timers. Timers include query encoding, retrieval, document fetch, token preparation, HTTP round trips, graph work and durable response logging; executor queue wait is excluded. Deadline checks occur between comparisons and do not establish a hard wall-clock cancellation guarantee.

Execution order uses `random.Random(20261111)` on all 450 query/method pairs, then skips saved pilot pairs. The first three new requests were run as a smoke batch before continuing the same order and ledger. Baseline timing repetition seeds are 20261111–20261113.

## Query identities

Evaluation IDs:

```text
1065, 316, 1238, 1161, 689, 225, 1227, 265, 364, 609, 748, 453, 1239, 490, 620, 848, 18, 228, 1212, 1102, 335, 1386, 290, 1269, 1309, 1046, 617, 310, 1186, 379, 1342, 611, 545, 283, 1033, 1003, 670, 1289, 407, 992, 488, 286, 340, 429, 801, 782, 788, 828, 588, 825, 505, 946, 149, 92, 355, 391, 515, 68, 222, 846, 209, 738, 63, 336, 211, 566, 224, 47, 815, 428, 345, 1136, 926, 61, 456, 1070, 679, 710, 1040, 434, 955, 1234, 897, 776, 1364, 474, 945, 739, 106, 1400, 877, 1050, 733, 1174, 772, 10, 255, 1043, 1037, 1390, 835, 1156, 978, 66, 1258, 1181, 671, 655, 657, 840, 787, 1304, 780, 1254, 1016, 1323, 746, 1247, 996, 929, 594, 200, 393, 542, 357, 1255, 531, 952, 959, 204, 119, 205, 740, 895, 1115, 1118, 1182, 82, 765, 244, 1376, 1006, 417, 876, 317, 242, 803, 1053, 256, 272
```

Tuning IDs:

```text
397, 1375, 259, 418, 1398, 12, 30, 114, 856, 1347, 1231, 486, 898, 930, 668, 818, 1367, 1311, 590, 646, 403, 220, 24, 1220, 795, 285, 663, 666, 470, 1268, 1322, 1153, 682, 1235, 1009, 1208, 1011, 886, 381, 251, 603, 1026, 970, 372, 246, 564, 576, 778, 553, 1158
```

## Environment

| Component | Version |
| --- | --- |
| Host | `macOS-27.2-arm64-arm-64bit` |
| Python | `3.12.13` |
| SQLite | `3.53.1` |
| sqlite-vec | `v0.1.10-alpha.4` |
| numpy | `2.5.4` |
| networkx | `3.7` |
| onnxruntime | `1.31.1` |
| tokenizers | `0.22.2` |
| tiktoken | `0.14.0` |

ONNX query sessions use four intra-operation threads and one inter-operation thread; BLAS uses one thread. SQLite cache is 2 MiB and mmap is disabled. Native baseline identity is the trusted `949073f-default` library; no dirty C source is compiled. The recovery/final archival ancestry is isolated from unrelated work at parent `812a5a38d710cdf492a430e85fcaf3da62ef1b4c`.

## Spending contract

Rates checked against the [official GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna) on 2026-11-11: $0.10/million uncached input tokens, $0.01/million cached input and $0.50/million output. Reasoning is an output subset. Dollar estimates are not a provider invoice.

Before each POST, reserve `(serialized UTF-8 payload bytes + 4096) * 125 + maximum completion tokens * 500` nanodollars. Refuse an input bound over 272,000 tokens or a reservation that would exceed the continuing $5 cap. Conservative settlement uses 125 nanodollars per input token (including a cache-write allowance) and 500 per output token. Unknown responses retain their entire reservation. Atomic ledger replacement syncs the file and containing directory.

## Input and source hashes

| Preserved original relative path | SHA256 |
| --- | --- |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/benchmark.py` | `334bd25224fe723b5305a84974fb39896e9831b32efdc2c80814ef9f5dd385e1` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/encoder.py` | `6860e52e291c470ce00ec24b94299e3c3639231a695296f72a3436163b0404c9` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/freeze.py` | `6f02aa447a81fb0cc69cd56525ef1a9ad263db3ea3d0d6d0180068fa54cb012b` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/build_baselines.py` | `f2cb8fe37a2daa1c24a9935922b2d322eccb02893b9da952a8f6b9d38fe34897` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/host_scorer.py` | `e6609b4f7a4802eeb0924e4e2fadad04668894f67b68b105e3936da0000c7d01` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/metrics.py` | `fb2d11c593d8d97d3b8a870c455007545a208ec93e3911b4065e1ed4063fa370` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/oracle.py` | `cf409fc1bbb130f415a3f409c52b1c1d388e5edfb43dd8803dd555554db13daa` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/recorded/protocol.json` | `278c9c65c90895fef0423b9afd17c5a66d635888e01f9b35a022c391f07243bf` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/recorded/score.json.gz` | `e26cf65a345eb5b5b0cf026a8a7d1da97894c1ad3234949f3f734be28336bf87` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/.work/dense-queries.npy` | `a00535ea987989804971c5aa5effab1faa2dd7007185d8b17cb26d5d553eb60b` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/.work/baseline-only.db` | `15e1220b8bd39eab4b86ec875dfc225d6e85b45a4fcdff69136b1b02f47ae9fa` |
| `benchmarks-ann/experiments/2026-10-26-scifact-hybrid/.work/scifact.zip` | `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` |
| `benchmarks-ann/experiments/2026-10-26-scifact-hybrid/.work/model.onnx` | `828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35` |
| `benchmarks-ann/experiments/2026-10-26-scifact-hybrid/.work/model-tokenizer.json` | `d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66` |
| `benchmarks-ann/experiments/2026-10-26-scifact-hybrid/.work/documents.npy` | `c2ae50d6b76c8a586c432c75ba964da6df92186aa82d7fce9a8409a28e9fd2cf` |
| `benchmarks-ann/experiments/2026-10-26-scifact-hybrid/.work/data-manifest.json` | `dc57edd8bb894642c9d653bea42fca6b912adb9402bf1f9345ac14a24f41e916` |
| `dist/bundles/949073f-default/lib/vec0.dylib` | `4785d6f2ef2b1da23ed8c99f218b9dd519f4270e0789a3a20d52f8727f4c0261` |

| Frozen executable source, relative to archive root | SHA256 |
| --- | --- |
| `2026-11-11-blitzrank/experiment.py` | `f9ebdbc3d79ea7ddd13fb45b44ecf664a2f27f6dee60006086924224d8aca6f6` |
| `2026-11-11-blitzrank/oracle.py` | `cd47ae37b81356dd259f450a9b2f380ff6f7ed2bd73ce3ee9cc3dac246d9567c` |
| `2026-11-11-blitzrank/.work/inputs/benchmarks-ann/experiments/2026-11-08-late-interaction/benchmark.py` | `334bd25224fe723b5305a84974fb39896e9831b32efdc2c80814ef9f5dd385e1` |
| `2026-11-11-blitzrank/.work/inputs/benchmarks-ann/experiments/2026-11-08-late-interaction/encoder.py` | `6860e52e291c470ce00ec24b94299e3c3639231a695296f72a3436163b0404c9` |
| `2026-11-11-blitzrank/.work/inputs/benchmarks-ann/experiments/2026-11-08-late-interaction/freeze.py` | `6f02aa447a81fb0cc69cd56525ef1a9ad263db3ea3d0d6d0180068fa54cb012b` |
| `2026-11-11-blitzrank/.work/inputs/benchmarks-ann/experiments/2026-11-08-late-interaction/build_baselines.py` | `f2cb8fe37a2daa1c24a9935922b2d322eccb02893b9da952a8f6b9d38fe34897` |
| `2026-11-11-blitzrank/.work/inputs/benchmarks-ann/experiments/2026-11-08-late-interaction/host_scorer.py` | `e6609b4f7a4802eeb0924e4e2fadad04668894f67b68b105e3936da0000c7d01` |
| `2026-11-11-blitzrank/.work/inputs/benchmarks-ann/experiments/2026-11-08-late-interaction/metrics.py` | `fb2d11c593d8d97d3b8a870c455007545a208ec93e3911b4065e1ed4063fa370` |

| Pinned upstream tournament module | SHA256 |
| --- | --- |
| `__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `compare_oracle.py` | `35f38cd2ca35ad11e631ab74c849fbbd0343e588e9e5063ebeb74d0a620ce75c` |
| `tournament_graph.py` | `7ecdb0267e6188c42305de7bc11782069efebcfa048e006e3a6242107e80fb26` |
| `tournament_graph_sort.py` | `aa2a77f5d63782761c179fbacd103d744e9f9819d2342198bf1a765fdd0c4563` |
| `types.py` | `89c94ba239720c4dedaf67f28210216949e4a568d82d99e6029925b223232351` |

The Git recovery snapshot retains raw inputs, candidate identities, model decisions, both protocols and the continuing ledger, plus only the helper sources and tokenizer data needed for offline replay. It excludes native libraries, databases, weights and virtual environments. [Data/upstream attribution](DATA-PROVENANCE.md) applies to the dataset-derived text and retained upstream components.

The full measurement outcome and verification are in [REPORT.md](REPORT.md). Reproduction instructions are in [README.md](README.md). The pilot outcome and original sources remain independently recoverable from the immutable recovery snapshot.
