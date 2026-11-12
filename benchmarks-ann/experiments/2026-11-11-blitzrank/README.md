# SciFact BlitzRank evaluation

The full 150-query evaluation is complete. RRF + BlitzRank reached 0.798189
nDCG@10 against 0.708804 for the tuning-selected dense baseline. The primary
gain is +0.089385, paired 95% interval [+0.042886, +0.138554], clearing the
unchanged relevance gate. All 450 query/method requests succeeded.

Total estimated API cost is $1.3865075; conservative accounting is $1.677428,
including the pilot, under the original $5 cap. The added 120-query batch used
three workers; its p95 request latency is about 19 seconds. No ranking default,
adapter, C-core, main-branch or CI change is made.

Read [the measured result](REPORT.md), [the full protocol](PROTOCOL.md) and
[data/upstream attribution](DATA-PROVENANCE.md). This is an expansion after an
observed pilot on previously examined public training queries, not a new blind
confirmatory benchmark.

## Replay without inference

The recovery snapshot contains exact executable sources, dependency versions,
450 receipts, both protocols, the continuing budget, pinned upstream tournament
modules, old helper source and the tokenizer cache. It excludes native libraries,
SQLite databases, model weights and virtual environments. Replaying does not
need the original host paths, original checkout, API key or inference service.

From a checkout containing the recovery Git object, use a fresh scratch directory:

```sh
RECOVERY_SHA=d502752999b6dad2812e89b85c801380676b499b
REPLAY=/tmp/sqlite-vec-blitzrank-replay
mkdir -p "$REPLAY"
git archive --format=tar --output "$REPLAY/source.tar" "$RECOVERY_SHA" \
  benchmarks-ann/experiments/2026-11-11-blitzrank
tar -xf "$REPLAY/source.tar" -C "$REPLAY"
EXPERIMENT="$REPLAY/benchmarks-ann/experiments/2026-11-11-blitzrank"
uv venv --python 3.12 "$REPLAY/venv"
uv pip install --python "$REPLAY/venv/bin/python" -r "$EXPERIMENT/requirements.lock"
"$REPLAY/venv/bin/python" -B -W error::ResourceWarning "$EXPERIMENT/offline_replay.py"
```

Dependency installation may access a package registry; use `--offline` with
cached pinned wheels when available. The replay itself removes `OPENAI_API_KEY`
and blocks Python connection/DNS calls. It verifies its 23 recovered helper
identities, runs ten contract tests and audits every actual model decision and
charge. Expected final output includes `verified:true`, 150 queries, 450
reranked requests, 1622 model calls and conservative USD 1.677428. The quality
gate passes, but a negative quality result would be distinct from failed
verification. The immutable recovery object is `d502752999b6dad2812e89b85c801380676b499b`; [recovery validation](RECOVERY.md) records the exercised Git-object replay.

Run from a writable disposable extraction, without Python `-O`. The verifier
creates the summary if absent and otherwise requires exact agreement. It never
makes model calls. Use a fresh directory rather than extracting over another
run or editing the frozen protocol.

## What was fixed

- Full 5,183-document SciFact corpus; original 50 tuning and 150 evaluation IDs.
- Exact BGE-small L2, FTS5 BM25 and RRF (constant 60). First 50 candidates per
  method, top ten output; no graph expansion or other reranking stage.
- Upstream BlitzRank revision `076018cf6153608203029e29562c36f50273b348`, window
  20, at most 50 rounds, exact upstream RankGPT prompt.
- OpenAI `gpt-6-luna`, standard tier, reasoning `low`, default temperature,
  2048 completion tokens including reasoning, no retries or model fallback.
- `o200k_base` prefix truncation at 1024 tokens per document/query. Malformed
  permutations reject the request instead of filling missing document IDs.

The primary target was RRF + BlitzRank versus the strongest tuning-selected
baseline: at least +0.02 nDCG@10 with a paired bootstrap lower bound above zero.
The gate and model settings were not changed after the pilot. The full protocol
was frozen before any added-query inference.

## Pilot reuse and spending

The original pilot contributes 90 requests and 322 model calls without rerunning
them. Its protocol, original receipts, code snapshots and ledger remain intact.
Each derived reused record carries its original path/SHA. Verification strips
only the documented extension binding and requires exact original equality.
The original ledger is an unchanged prefix of the full ledger.

There is one $5 cap. Before every request, reserve one input token per serialized
UTF-8 byte plus 4096 framing tokens, priced at $0.125/million, and the full output
allowance at $0.50/million. Reject input bounds over 272000 tokens and calls that
cannot fit the remaining cap. Unknown responses retain their full reservation.
Locked settlement and atomic file/directory sync protect the shared ledger.
Reasoning tokens are not added twice. Dollar estimates are not invoices.

The runner has a process lock and one shared budget across worker threads. Each
worker has its own SQLite connection and BGE session. The 20-million-token run
admission remains in force. `experiment.py run` skips saved requests, including
failures; it does not silently retry or replace them. An interrupted
`.calls.partial` journal blocks automatic continuation of that request.

## Repeat live retrieval or inference

Repeating paid inference is separate from replaying receipts and needs explicit
new authorization. Do not reset the ledger or treat the cap as a per-run budget.
Restore or regenerate the original large assets named in the input inventory,
then freeze a fresh protocol with their actual identities. The recovery snapshot
has only the small source/text assets needed for offline verification.

Historical prerequisite setup is available from Git snapshot
`4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd`, particularly the SciFact hybrid and
late-interaction READMEs. Ignored native binaries/databases/weights are not
recoverable from Git alone. New toolchains or regenerated databases may require
new provenance instead of claiming exact-byte reproduction. Full live execution
was measured on macOS arm64; other live platforms remain unqualified.

## Limits and archive lifecycle

This is one model alias, window and fixed candidate budget with one inference
pass. The 30-query pilot and 120-query extension have separate latency strata;
there is no uniform sequential p95 over all 150, concurrency load sweep,
production-serving qualification, cross-corpus result or agent answer-quality
claim. Cyclic model preferences frequently create tied tiers.

The executable recovery snapshot is preserved for independent review. The current archive retains Markdown only and links to that immutable snapshot.
Original active scratch inputs remain protected separately; no shared cache or
unrelated working-tree change is discarded. Integration review owns publication
to the development branch and wiki. No push occurs from this experiment chat.
