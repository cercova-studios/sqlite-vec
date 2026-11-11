# Scoped hybrid retrieval evaluation

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

Compare vector-only, FTS5-only and the published scoped RRF query using explicit
synthetic relevance judgments and an independent eligibility model. See the
[report](REPORT.md) for recorded results and limits. The evaluation changes no
engine behavior or retrieval defaults.

From the repository root on macOS arm64, with the project's `uv` test environment
and build prerequisites available:

```sh
make loadable
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python benchmarks-ann/experiments/2026-10-26-scoped-hybrid/test_evaluate.py -v
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python benchmarks-ann/experiments/2026-10-26-scoped-hybrid/evaluate.py --output benchmarks-ann/experiments/2026-10-26-scoped-hybrid/.work
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python benchmarks-ann/experiments/2026-10-26-scoped-hybrid/verify.py
```

Fresh results go to ignored `.work/`; the final command verifies the frozen
`recorded/` evidence. Use `--extension /absolute/path/to/vec0.dylib` to benchmark
a specific library. Set `SCOPED_HYBRID_EXTENSION` to the same path when running
the tests. Replay source/extension hashes can differ from the historical run.
Run Python with assertions enabled. No extra Python dependencies are required.

`fixture.json` specifies sixteen actionable procedures, nine queries, scope and
validity boundaries, relevance labels, an invalidation and a replacement update.
Four-dimensional vectors are authored coordinates, not learned embeddings.
Distractors have no relevant labels. Their attributes/text/vectors are identical
across seeds; only insertion order changes. Four of each ten distractors belong
to another tenant, two expire at time 100, two start at 101, one is invalidated,
and one remains eligible at time 100. Topic selection is seeded independently
of insertion order. Earlier/later queries deliberately change eligibility.

The default sweep uses 0/1,000/10,000 distractors, insertion seeds 0/1/2,
branch limits 10/50 and nine timing rounds after two warmups. Each database is
queried at baseline, after committed invalidation, after a committed atomic
scope/text/vector replacement and after reopen. Vector-only and FTS-only execute
their full candidate budgets before Python selects ten results; hybrid returns
ten in SQL. This measures matched branch budgets, not optimized single-branch
queries with an early LIMIT 10.

The harness extracts the current scoped query from `site/guides/hybrid-search.md`.
Recorded guide bytes and hashes preserve the measured version. Vector results
are checked against direct float32-input/Python-distance exhaustive ranking,
including tied cutoffs. Fusion ranks, scores and source IDs are checked against
separate branch queries and Python RRF. The fixture's relevance labels are not
derived from those ranks. `verify.py` regenerates relevance, eligibility and
summary metrics from the saved result IDs and latencies without rerunning SQL.

Tests check hand-specified boundary sets, pending writes, savepoint/full rollback,
commit/reopen, scope/text/vector consistency and the late-filter control. Removing
tenant, invalidation or either validity boundary must make the oracle reject
results. These checks validate the evaluation; they do not establish production
relevance, authorization or downstream agent usefulness.
