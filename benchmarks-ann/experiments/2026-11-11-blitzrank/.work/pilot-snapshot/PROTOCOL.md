# Frozen BlitzRank protocol and provenance

Frozen before hosted inference on 2026-11-11T18:57:10Z. The measured result is in [REPORT.md](REPORT.md); runnable commands are in [README.md](README.md).

## Fixed comparison

- Corpus: BEIR SciFact, 5,183 documents; title + single space + text.
- Dense model: `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, 384 normalized float32 dimensions; exact L2.
- Query instruction: `Represent this sentence for searching relevant passages: ` (including the trailing space).
- BM25: SQLite FTS5, natural-language claim tokenized into quoted terms joined by OR.
- RRF: top 50 from each branch, constant 60, score descending then integer document ID; rerank the first 50 fused documents.
- BM25/dense each retrieve 100, then admit the first 50 to reranking; top-ten baseline order remains unchanged.
- Dense won the original 50-query tuning comparison. Primary outcome: RRF + BlitzRank minus dense nDCG@10.
- Quality gate: +0.02 absolute nDCG@10 and paired 95% bootstrap lower bound above zero; 10,000 resamples, seed 20261111.
- No performance/adoption gate was calibrated for this hosted-model pilot.

## Model and request contract

- OpenAI `gpt-6-luna` alias, standard service tier, Chat Completions at `https://api.openai.com/v1/chat/completions`.
- Existing process `OPENAI_API_KEY` reused with explicit human authorization. Its value is never retained.
- Reasoning effort `low`; temperature omitted; completion cap 2,048 including reasoning; `store:false`; no tools.
- Window 20, top ten, at most 50 tournament rounds. No retries or model fallbacks.
- Exact upstream RankGPT prompt; `o200k_base` prefix truncation at 1,024 tokens for each query/document.
- A complete permutation is mandatory. Missing, repeated, out-of-range or malformed IDs reject the request instead of repairing the response.
- HTTP timeout 120 seconds per call; request deadline checked between comparisons at 1,800 seconds. This is not a hard wall-clock cancellation guarantee.
- Run token admission: 20,000,000 tokens. Monetary cap: $5 across all hosted calls in this ledger.
- Before POST, reserve `(UTF-8 payload bytes + 4096) * 125 + maximum completion tokens * 500` nanodollars. Reject input bounds over 272,000 tokens.
- Settle at 125 nanodollars per input token and 500 per output token; preserve the whole reservation after an unknown response. Standard-price estimates use 100/10/500 for uncached/cached/output tokens.
- A process lock excludes overlapping run commands. Record each received response before parsing; retain interruption journals.

## Queries and timing

This reuses public training annotations and previously examined queries. The 30-query subset was selected as the earlier experiment's timing sample before BlitzRank outcomes; it is not a fresh blind holdout.

Evaluation query IDs:

```text
828, 1239, 1255, 456, 1003, 657, 10, 788, 1238, 929, 1309, 149, 531, 355, 876, 286, 1234, 710, 1115, 772, 317, 978, 391, 803, 434, 92, 787, 1400, 1390, 453
```

Tuning query IDs:

```text
397, 1375, 259, 418, 1398, 12, 30, 114, 856, 1347, 1231, 486, 898, 930, 668, 818, 1367, 1311, 590, 646, 403, 220, 24, 1220, 795, 285, 663, 666, 470, 1268, 1322, 1153, 682, 1235, 1009, 1208, 1011, 886, 381, 251, 603, 1026, 970, 372, 246, 564, 576, 778, 553, 1158
```

Method/query execution order: shuffle the 90 evaluation pairs with Python `random.Random(20261111)`. Baseline timing has three shuffled repetitions (seeds 20261111, 20261112, 20261113); reranking has one pass.

## Environment

| Component | Measured version |
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

ONNX query inference uses four intra-operation threads and one inter-operation thread; BLAS uses one. SQLite cache is 2 MiB; mmap is disabled. Local BGE initialization/warmup is outside timers. Document vectors originated under ONNX Runtime 1.23.2; live query inference here uses 1.31.1 and checks exact candidate parity.

The native library is the trusted prebuilt `949073f-default` artifact, not current dirty C sources. The executable experiment sources are currently uncommitted and identified by hashes below. The prior scripts/receipts are recoverable from `4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd`.

## Preserved input and helper identities

Every copy was verified before releasing the older experiment paths for teardown. Paths below are relative to `.work/inputs/`, which preserves their old repository-relative layout. Exact-byte regeneration of ignored databases/native binaries is not guaranteed; retain the copies or freeze new provenance explicitly.

| Original relative path | Bytes | SHA256 |
| --- | ---: | --- |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/benchmark.py` | 11776 | `334bd25224fe723b5305a84974fb39896e9831b32efdc2c80814ef9f5dd385e1` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/encoder.py` | 6700 | `6860e52e291c470ce00ec24b94299e3c3639231a695296f72a3436163b0404c9` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/freeze.py` | 5712 | `6f02aa447a81fb0cc69cd56525ef1a9ad263db3ea3d0d6d0180068fa54cb012b` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/build_baselines.py` | 3606 | `f2cb8fe37a2daa1c24a9935922b2d322eccb02893b9da952a8f6b9d38fe34897` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/host_scorer.py` | 6595 | `e6609b4f7a4802eeb0924e4e2fadad04668894f67b68b105e3936da0000c7d01` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/metrics.py` | 1570 | `fb2d11c593d8d97d3b8a870c455007545a208ec93e3911b4065e1ed4063fa370` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/oracle.py` | 4824 | `cf409fc1bbb130f415a3f409c52b1c1d388e5edfb43dd8803dd555554db13daa` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/recorded/protocol.json` | 9317 | `278c9c65c90895fef0423b9afd17c5a66d635888e01f9b35a022c391f07243bf` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/recorded/score.json.gz` | 733236 | `e26cf65a345eb5b5b0cf026a8a7d1da97894c1ad3234949f3f734be28336bf87` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/.work/dense-queries.npy` | 307328 | `a00535ea987989804971c5aa5effab1faa2dd7007185d8b17cb26d5d553eb60b` |
| `benchmarks-ann/experiments/2026-11-08-late-interaction/.work/baseline-only.db` | 33976320 | `15e1220b8bd39eab4b86ec875dfc225d6e85b45a4fcdff69136b1b02f47ae9fa` |
| `benchmarks-ann/experiments/2026-10-26-scifact-hybrid/.work/scifact.zip` | 2816079 | `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` |
| `benchmarks-ann/experiments/2026-10-26-scifact-hybrid/.work/model.onnx` | 133093490 | `828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35` |
| `benchmarks-ann/experiments/2026-10-26-scifact-hybrid/.work/model-tokenizer.json` | 711396 | `d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66` |
| `benchmarks-ann/experiments/2026-10-26-scifact-hybrid/.work/documents.npy` | 7961216 | `c2ae50d6b76c8a586c432c75ba964da6df92186aa82d7fce9a8409a28e9fd2cf` |
| `benchmarks-ann/experiments/2026-10-26-scifact-hybrid/.work/data-manifest.json` | 1773 | `dc57edd8bb894642c9d653bea42fca6b912adb9402bf1f9345ac14a24f41e916` |
| `dist/bundles/949073f-default/lib/vec0.dylib` | 282072 | `4785d6f2ef2b1da23ed8c99f218b9dd519f4270e0789a3a20d52f8727f4c0261` |

## Frozen executable source identities

| Source relative to the experiment archive root | SHA256 |
| --- | --- |
| `2026-11-11-blitzrank/experiment.py` | `a314d03944d4c9e5370121c08e62a4578e7b459b71b08c9c2a2003aa8a152cfe` |
| `2026-11-11-blitzrank/oracle.py` | `89b7bb23fcc9c3026824ed6a8f28415e053b52f3e3414909e8e5d6b345cb3adb` |
| `2026-11-11-blitzrank/.work/inputs/benchmarks-ann/experiments/2026-11-08-late-interaction/benchmark.py` | `334bd25224fe723b5305a84974fb39896e9831b32efdc2c80814ef9f5dd385e1` |
| `2026-11-11-blitzrank/.work/inputs/benchmarks-ann/experiments/2026-11-08-late-interaction/encoder.py` | `6860e52e291c470ce00ec24b94299e3c3639231a695296f72a3436163b0404c9` |
| `2026-11-11-blitzrank/.work/inputs/benchmarks-ann/experiments/2026-11-08-late-interaction/freeze.py` | `6f02aa447a81fb0cc69cd56525ef1a9ad263db3ea3d0d6d0180068fa54cb012b` |
| `2026-11-11-blitzrank/.work/inputs/benchmarks-ann/experiments/2026-11-08-late-interaction/build_baselines.py` | `f2cb8fe37a2daa1c24a9935922b2d322eccb02893b9da952a8f6b9d38fe34897` |
| `2026-11-11-blitzrank/.work/inputs/benchmarks-ann/experiments/2026-11-08-late-interaction/host_scorer.py` | `e6609b4f7a4802eeb0924e4e2fadad04668894f67b68b105e3936da0000c7d01` |
| `2026-11-11-blitzrank/.work/inputs/benchmarks-ann/experiments/2026-11-08-late-interaction/metrics.py` | `fb2d11c593d8d97d3b8a870c455007545a208ec93e3911b4065e1ed4063fa370` |

Upstream source: [ContextualAI/BlitzRank](https://github.com/ContextualAI/BlitzRank/tree/076018cf6153608203029e29562c36f50273b348), revision `076018cf6153608203029e29562c36f50273b348`. Imported tournament modules remain unchanged.

| Upstream tournament module | SHA256 |
| --- | --- |
| `__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `compare_oracle.py` | `35f38cd2ca35ad11e631ab74c849fbbd0343e588e9e5063ebeb74d0a620ce75c` |
| `tournament_graph.py` | `7ecdb0267e6188c42305de7bc11782069efebcfa048e006e3a6242107e80fb26` |
| `tournament_graph_sort.py` | `aa2a77f5d63782761c179fbacd103d744e9f9819d2342198bf1a765fdd0c4563` |
| `types.py` | `89c94ba239720c4dedaf67f28210216949e4a568d82d99e6029925b223232351` |

## Retained record identities

| Record | SHA256 |
| --- | --- |
| `recorded/protocol.json` | `63febb9b5c5eb10cf9d51f13382c145b637b6355d0874739537c193c51aaa464` |
| `recorded/inputs.json.gz` | `b934a4b1614e5d8851a88b89b79f0ba3bdcc8a8b74a7ed7344df9156997a2b50` |
| `recorded/baselines.json.gz` | `c6aa7df1ae3c3412009a7f2c66b9424964f0ffd2464bc24dae6c360430163cdc` |
| `recorded/input-inventory.json` | `1d48632a3be9eacce3894c6d2c32668b2dde971f9b327d5a48925ca126b05cfb` |
| `recorded/summary.json` | `971085b8ddea9defb60803a51ec75ab6c558bf32484d50e56018c4def256e54e` |
| `recorded/budget.json` | `5286cd60576edb9c57f0a01a67e4e52ead49d9b90683d54f7e113b35b0fa337f` |
| `recorded/dense-oracle.json` | `10b534cf9c33454f4ac14cc9330aa116ee8e575ebfb1e32d8c1cad0f3d8311ea` |
| `recorded/tests.log` | `397773abe9dd51fc706042ee405b60d74fcab53b595029d44e48d2955b28660b` |

The final manifest also hashes the current documentation, verifier, test harness and all 90 compressed response receipts. It excludes itself and ignored `.work/` assets. This is an integrity record, not authentication against a hostile writer.

## Reproduction and retirement

Use the commands in [README.md](README.md) to verify the saved run without API access. `verify.py` replays model decisions locally; it does not call the model. A repeated live experiment requires a fresh protocol/destination and separate paid-call authorization.

Keep this uncommitted review workspace protected until an authorized Git recovery commit or explicit backup/disposal decision. Measurement is complete; retirement remains open. No wiki publication or deletion of unique run data is authorized by this report.
