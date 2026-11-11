# Late-interaction reference and SciFact experiment

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

The scalar reference and measured SciFact experiment are complete. The frozen
reranker reduced relevance, so native MaxSim was rejected for this configuration.
Read [the result](REPORT.md), [specification](SPEC.md) or
[machine-readable summary](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-late-interaction/recorded/summary.json).

## Verify the retained result offline

From the repository root, Python 3.12 or newer can check hashes, coverage,
independent relevance metrics, work accounting and live/precomputed ranking parity
without model weights, NumPy, a native extension or network access:

```sh
python3 benchmarks-ann/experiments/2026-11-08-late-interaction/analyze.py
```

The command prints the measured summary with `all_gates_pass: false`. That is the
expected negative research outcome; verification itself exits successfully.
`--database PATH` additionally verifies the original combined SQLite file hash.
`--write-summary` regenerates `recorded/summary.json`; it does not run inference.
Use normal Python execution, without `-O`, so executable assertions remain enabled.

The reference-only tests need stdlib Python:

```sh
python3 -W error::ResourceWarning -m unittest discover \
  -s benchmarks-ann/experiments/2026-11-08-late-interaction -p test_oracle.py -v
```

With NumPy installed, run all 23 tests:

```sh
python3 -W error::ResourceWarning -m unittest discover \
  -s benchmarks-ann/experiments/2026-11-08-late-interaction -p 'test_*.py' -v
```

## Repeat the live experiment

This procedure explicitly downloads public model data and runs CPU inference.
The recorded measurements used an isolated Python 3.12 environment with the exact
versions in `requirements.lock`; no repository/global environment was changed.
The extension-loading boundary trusts the exact locally built library hash,
not arbitrary downloaded native code.

1. Supply the existing assets from the
   [earlier SciFact experiment](../2026-10-26-scifact-hybrid/README.md) in its
   `.work/`: `scifact.zip`, `documents.npy`, `model.onnx`,
   `model-tokenizer.json` and `data-manifest.json`. Their required identities are
   pinned here. Restore/build the `949073f-default` bundle with the library hash
   in `build_baselines.py`; another build is a new provenance choice.
2. Create an isolated Python 3.12 environment and install `requirements.lock`.
   The recorded wheels were macOS arm64; other platforms may need different
   runtime versions and must retain that difference.
3. Run `acquire.py`, then `freeze.py`, then `prepare.py`, then
   `build_baselines.py` from this directory using that Python. Acquisition uses
   network access; the remaining stages use local data. Preserve the four
   preparation scripts: their exact hashes are verified before measurement.
4. Run `run_experiment.py score`, followed by `run_experiment.py live`.
   The controller sets offline-model and thread environment variables and
   records true fresh-child peak memory. Scoring itself uses precomputed
   embeddings; the separate, explicit live phase invokes both encoders.

Preparation requires a fresh `.work/corpus.db`; baseline construction also
requires a fresh `.work/baseline-only.db`. Measurement phases require fresh
output/log paths and scoring requires a fresh `.work/selection.json`. These
checks prevent silent overwrite. Move a previous attempt aside with its logs
and script versions before repeating. Do not alter the committed frozen protocol
to fit a result. A changed model/split/budget belongs in a new experiment archive.

The large `.work/` assets are ignored. `recorded/` retains the actual quality,
timing, resource and provenance receipts from this run. No production schema,
writer history, CLI/MCP operation or SQLite function was changed. Native
cancellation, managed artifact publication and update/storage churn remain later
slices, contingent on a useful relevance result.
