# Replay the comparison follow-up

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

Read [the findings and limitations](REPORT.md) first. These scripts measure SQL filter construction, flat BLOB access, minimum-hop graphs and Turso/libSQL vectors. They do not change the working C core or application dependencies.

## Verify the saved archive

From the repository root, with Python 3 and no engine packages:

```sh
python3 benchmarks-ann/experiments/2026-11-03-comparison-followup/verify.py
```

Expected: `Verified 2200 quality points, 6600 timed calls, diagnostic summaries and corruption rejection`. The verifier uses the prior comparison's tracked frozen oracle and the current G0 fixtures. It checks archive hashes and recomputes summaries. It does not rerun engines or independently recompute scores without the original vector inputs.

## Rerun measurements

Use macOS arm64, a C compiler and the repository's vendor SQLite headers for the recorded M1/NEON compile flags. First follow the [previous experiment's input/build instructions](../2026-11-03-engine-comparison/README.md) to regenerate `.work/build/vec0.dylib` and both datasets. Match its recorded source/input hashes. Generated libraries and databases stay under ignored `.work/`; do not use a dirty production source snapshot.

Install the benchmark packages in an isolated environment, not the application environment:

```sh
python3.12 -m venv /tmp/vec-comparison-followup-venv
/tmp/vec-comparison-followup-venv/bin/pip install numpy==2.5.4 pyturso==0.8.3 libsql==0.1.11 ladybug==0.22.1
```

Run each command from the repository root. These commands replace recorded receipts; use a separate checkout to preserve the checked-in evidence. `profile_filter.py` refuses an existing `.work/filter.db`; start with a fresh experiment work directory. Later SQL controls require the database it creates. For comparable new timings, stop other benchmark jobs and serialize these commands.

```sh
/tmp/vec-comparison-followup-venv/bin/python benchmarks-ann/experiments/2026-11-03-comparison-followup/profile_filter.py
/tmp/vec-comparison-followup-venv/bin/python benchmarks-ann/experiments/2026-11-03-comparison-followup/sql_controls.py
/tmp/vec-comparison-followup-venv/bin/python benchmarks-ann/experiments/2026-11-03-comparison-followup/flat_reads.py
/tmp/vec-comparison-followup-venv/bin/python benchmarks-ann/experiments/2026-11-03-comparison-followup/graph_compare.py sqlite
/tmp/vec-comparison-followup-venv/bin/python benchmarks-ann/experiments/2026-11-03-comparison-followup/graph_compare.py ladybug
/tmp/vec-comparison-followup-venv/bin/python benchmarks-ann/experiments/2026-11-03-comparison-followup/graph_compare.py sqlite --deep
/tmp/vec-comparison-followup-venv/bin/python benchmarks-ann/experiments/2026-11-03-comparison-followup/graph_compare.py ladybug --deep
/tmp/vec-comparison-followup-venv/bin/python benchmarks-ann/experiments/2026-11-03-comparison-followup/sql_engines.py libsql scifact
/tmp/vec-comparison-followup-venv/bin/python benchmarks-ann/experiments/2026-11-03-comparison-followup/sql_engines.py libsql sift50k
/tmp/vec-comparison-followup-venv/bin/python benchmarks-ann/experiments/2026-11-03-comparison-followup/sql_engines.py turso scifact
/tmp/vec-comparison-followup-venv/bin/python benchmarks-ann/experiments/2026-11-03-comparison-followup/sql_engines.py turso sift50k
/tmp/vec-comparison-followup-venv/bin/python benchmarks-ann/experiments/2026-11-03-comparison-followup/transaction_probe.py
```

The recorded Ladybug deep probe fails at its 64 MiB buffer-pool limit. The wrapper records that failure with `status=failed`; process exit alone is not a pass. Turso score checks are reported as failed flags while retaining diagnostics. Other validation failures raise exceptions. Inspect receipts before using results.

`verify.py --write` regenerates summaries/hashes after reviewed evidence changes; it is not a substitute for passing live checks. `recorded/runtime.json` pins the measured environment, previous input/oracle digests, fixture digests and diagnostic source digests. The original full package/hardware manifest remains in the prior experiment. Update provenance deliberately for a new campaign. No model downloads or external services are needed after regenerating the previous public inputs.
