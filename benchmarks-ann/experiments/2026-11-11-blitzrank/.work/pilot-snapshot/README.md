# SciFact BlitzRank comparison

Compare BM25, exact dense retrieval and reciprocal rank fusion (RRF) with the
same candidates reranked by BlitzRank using OpenAI `gpt-6-luna`. This independent
experiment changes neither the search adapter nor the SQLite extension.

The frozen run uses 30 previously selected evaluation queries, 50 candidates
per query/method, and top-10 results. It is a local pilot on public SciFact
training annotations, not a fresh blind benchmark or a reproduction of the
BlitzRank paper's model results. Read [the report](REPORT.md) for results and
limitations after the run completes.

## Method

- Reuse the 5,183-document SciFact corpus, BGE-small dense representations and
  query split from the earlier late-interaction experiment. Recompute all 600
  baseline query/method rankings and require exact candidate identity agreement
  with its retained receipts before selecting a reranking shortlist.
- Select the strongest baseline on the original 50 tuning queries. Evaluate
  reranking on the original experiment's 30 timing queries. Their identities were
  selected before any BlitzRank result. The remaining 120 evaluation queries are
  outside this pilot.
- Keep the first 50 candidates from each baseline. Dense and BM25 originally
  return 100; RRF merges the first 50 from each branch with constant 60. Report
  candidate recall for the actual 50-document shortlist.
- Use upstream BlitzRank revision
  `076018cf6153608203029e29562c36f50273b348`, window 20, top 10 and at most
  50 rounds. Load its unchanged tournament modules directly; the full upstream
  package and its unrelated evaluation/model SDK dependencies are unnecessary.
- Use the upstream RankGPT conversational prompt, `o200k_base` truncation at
  1,024 tokens per document/query, reasoning effort `low`, default temperature,
  and a 2,048-token completion limit including reasoning. Reject malformed
  permutations rather than appending missing identifiers. Make no automatic
  retries, model substitutions or baseline fallbacks.

The primary comparison is RRF + BlitzRank against the tuning-selected baseline.
The predeclared relevance target is +0.02 absolute nDCG@10 with a paired 95%
query-bootstrap interval above zero (10,000 samples, seed 20261111). Other
same-candidate comparisons are descriptive. No production latency target is
claimed for this exploratory hosted-model run.

## Inputs and isolation

`recorded/protocol.json` freezes identities, budgets, settings, script hashes,
upstream hashes and environment versions before hosted inference.
`recorded/input-inventory.json` lists the 17 preserved source/assets and hashes.
The active copies live under `.work/inputs/` in their original repository-relative
tree. The runner imports only those copies; teardown of older experiments does
not affect this experiment.

Large dataset/model/database inputs, the upstream checkout and virtual environment
are ignored. The exact baseline library comes from the older `949073f` build;
no current working-tree C source is compiled or loaded. The baseline database
opens immutable and read-only after rejecting WAL/journal files and verifying
its integrity and complete source text against the frozen public corpus.

## Run with the preserved inputs

Run from the repository root on the measured macOS arm64 host. The existing
`OPENAI_API_KEY` must be present in the process environment. Do not paste it into
commands, output, receipts or documentation. The user authorized reuse of that
key, model `gpt-6-luna`, and a total experiment cap of $5.

```sh
EXPERIMENT=benchmarks-ann/experiments/2026-11-11-blitzrank
"$EXPERIMENT/.work/venv/bin/python" -W error::ResourceWarning \
  -m unittest discover -s "$EXPERIMENT" -p 'test_*.py' -v
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  "$EXPERIMENT/.work/venv/bin/python" "$EXPERIMENT/experiment.py" run
"$EXPERIMENT/.work/venv/bin/python" "$EXPERIMENT/experiment.py" analyze
"$EXPERIMENT/.work/venv/bin/python" "$EXPERIMENT/verify.py"
```

`run` resumes by skipping complete saved requests, including failed requests;
it never silently replaces outcomes. `--max-tasks 1` executes at most one unsaved
request without changing the frozen evaluation domain. `analyze` exits 1 when
coverage is incomplete or requests failed. Verification errors also fail the
command; a negative relevance result is distinct from failed verification.

Each finished request is an exclusive compressed JSON receipt with candidates,
ranked IDs, raw prompts/responses, token usage, round schedules, SCC statistics
and elapsed times. Analysis replays the pinned upstream graph from the actual
recorded comparisons and requires the same schedule, top-10 IDs and call count.

The append-only `.calls.partial` journal retains received responses during an
interruption. A partial journal blocks automatic continuation of that request;
inspect it and the budget ledger before creating a separately identified retry.
Do not delete the journal or reset the ledger to obtain a passing result.

## Reproduce in a fresh environment

The measured dependencies are pinned in `requirements.lock`. Create an isolated
Python 3.12 environment under `.work/venv` and install that lock. Clone
[upstream BlitzRank](https://github.com/ContextualAI/BlitzRank) into
`.work/upstream` and check out the exact revision above. Its MIT license is
retained in `UPSTREAM-LICENSE`, including the prompt attribution.

The older executable sources and receipts are recoverable from repository Git
snapshot `4c5a5d1`. Restore that snapshot into a separate checkout and follow its
`2026-11-08-late-interaction/README.md` input preparation instructions. The
SciFact dataset and BGE assets originated in its
`2026-10-26-scifact-hybrid/.work/`. Copy the files listed in
`recorded/input-inventory.json` from that checkout into this experiment's
`.work/inputs/`, preserving the listed relative paths, and verify every hash.
Ignored binaries and databases are not recoverable from Git alone; regenerated
assets may require a newly frozen provenance record rather than claiming an
exact-byte reproduction.

Before a new hosted run, retain the current receipts and ledger in a separate
run directory and use a fresh `recorded/` destination. `experiment.py prepare`
reruns baseline quality and live timing, checks preserved inputs, and freezes the
new protocol. It rejects an existing destination. Its default is the 30-query
pilot; `--evaluation-count 150` selects the full prior evaluation split before
inference. Do not alter a frozen protocol after observing outcomes. Explicitly
authorize a new spending cap before making additional paid calls.

## Latency, tokens and spending

Baseline latency includes live BGE query encoding where applicable and exact
SQLite retrieval. There are three repetitions per query/method after warmup.
BlitzRank latency is one measured pass per query/method and includes encoding,
retrieval, text fetch, prompt preparation, HTTP requests, graph work and durable
receipt logging. Model load and input verification are outside both timers.
No filesystem cache flush, concurrent-serving test or latency repetition is
claimed for reranking. The API model alias may change behind the same name.

Input and output token usage comes from each provider response. Reasoning tokens
are a subset of output tokens and are never added twice. Standard-price dollar
estimates use $0.10 per million uncached input tokens, $0.01 per million cached
input tokens and $0.50 per million output tokens, checked on 2026-11-11 against
the [official model page](https://developers.openai.com/api/docs/models/gpt-6-luna).
Estimates are not a provider invoice. Baselines make no paid model calls; local
embedding compute and electricity costs are not priced.

Before each API request, the durable budget reserves an input bound of one token
per serialized UTF-8 byte plus 4,096 framing tokens, and the full output allowance.
The input reserve uses $0.125 per million tokens to include a cache-write premium;
output uses $0.50. Prompts whose bound exceeds 272,000 tokens are rejected to
avoid long-context pricing. Standard service tier, text-only requests and no
tools keep the charge model bounded. Received usage settles the reservation at
the conservative rates. A lost response retains its full reservation. The runner
rejects a call if the remaining $5 cannot cover its reservation, and a process
lock excludes overlapping runs. The cap covers this experiment's calls only.

## Validation and calibration

Tests cover complete permutations, malformed generations, missing/duplicate IDs,
reasoning-token accounting, budget refusal before inference, durable reservations,
exact upstream prompt parity, the actual sorter with an independent known-order
oracle, and refusal to accept successful receipts with no model calls.
Three tuning queries also check 150 native L2 distances against independent
float64 NumPy calculations, including candidate cutoff coverage. The observations
are retained in `recorded/dense-oracle.json`.

Before the hosted-model choice, one local GPT-OSS-20B calibration request returned
three identifiers for twenty documents. It was recorded as failed without repair
or retry; its 8,036 tokens were local and incurred no API charge. That calibration
and its original source snapshot are isolated in `.work/local-calibration/` and
are excluded from all hosted results. The task-owned local model and loopback
server were stopped after the user selected hosted inference.
