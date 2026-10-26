# Experimental HNSW alpha pruning — 2026-10-26

**Decision: keep α=1.0.** The real-data measurements do not support changing
production pruning. See [the report](REPORT.md) for recall/QPS tables, plots,
methodology and limits. This archive adds no SQL option or production code.

The experiment used GloVe-100 cosine and SIFT-128 L2 subsets (50,000 vectors,
1,000 held-out queries each), four alpha values, three insertion orders, and
`ef_search=32/64/128/256`. A separate alternating alpha-1/1.1 timing check
confirmed that the earlier synthetic speedup did not generalize.

## Recorded evidence

- [results/summary.csv](results/summary.csv) and JSON contain the full sweep.
- [results/paired_summary.json](results/paired_summary.json) contains the alternating check.
- `results/raw/*.json.gz` and `results/paired_timing.json.gz` preserve per-query
  measurements with lossless gzip compression. There are 640,000 timed queries.
- [provenance/build-manifest.json](provenance/build-manifest.json) records source
  hashes, compiler flags, library hashes, HEAD and the pre-existing working diff.
- [provenance/data-manifest.json](provenance/data-manifest.json) records public
  URLs, complete dataset hashes and selected-array hashes. Dataset files and
  compiled libraries are excluded from Git.
- `recorded/scripts/` and [recorded/REPORT-original.md](recorded/REPORT-original.md)
  preserve the original scripts and report verbatim, including historical paths.
  Raw receipts identify the original benchmark script by SHA256.
- `scripts/` contains runnable copies with checkout-relative paths. `manifest.json`
  hashes every archived file except itself. It also records uncompressed hashes.

## Verify the archive

From the repository root, run:

```sh
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python benchmarks-ann/experiments/2026-10-26-hnsw-alpha/verify.py
```

This checks file hashes, script provenance, record counts and library identities,
then regenerates both summaries from the archived raw receipts and compares them
with the saved summaries. It needs NumPy (recorded version 2.1.3), but downloads
no datasets and runs no graph benchmark. Run Python with assertions enabled.

## Run a new comparison

The runnable copies target macOS arm64, `cc`, Apple M1 code generation and NEON.
They use the current checkout, including uncommitted C changes, and record its
HEAD and working diff. They do not reconstruct the old source automatically.
The recorded baseline was `c85e736b2e7ef008fbfac0733b0de96ad7ed99ef` plus the
helper diff in the build manifest. Exact historical reconstruction also requires
matching every recorded source hash, including generated headers.

Run these commands from the repository root. `uv`, the repository test environment,
SQLite headers under `vendor/`, and the generated `sqlite-vec.h` are prerequisites.
Use a fresh output directory: the sweep resumes existing results by script hash.
The default output `.work/` is ignored; `HNSW_ALPHA_WORKDIR` selects another directory.
Never use the frozen `results/` directory as a work directory.

```sh
export HNSW_ALPHA_WORKDIR="$PWD/benchmarks-ann/experiments/2026-10-26-hnsw-alpha/.work"
export UV_CACHE_DIR=/tmp/uv-cache-muse
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
experiment=benchmarks-ann/experiments/2026-10-26-hnsw-alpha
make loadable
uv run --managed-python --project tests python "$experiment/scripts/build_variants.py"
curl --fail --location https://ann-benchmarks.com/glove-100-angular.hdf5 --output "$HNSW_ALPHA_WORKDIR/data/glove-100-angular.hdf5"
curl --fail --location https://ann-benchmarks.com/sift-128-euclidean.hdf5 --output "$HNSW_ALPHA_WORKDIR/data/sift-128-euclidean.hdf5"
cc -O3 -fPIC -shared "$experiment/scripts/exact_oracle.c" -o "$HNSW_ALPHA_WORKDIR/exact_oracle.dylib"
uv run --managed-python --project tests --with h5py==3.16.0 python "$experiment/scripts/prepare_data.py"
uv run --managed-python --project tests python "$experiment/scripts/benchmark.py" validate
uv run --managed-python --project tests python "$experiment/scripts/benchmark.py" sweep
uv run --managed-python --project tests python "$experiment/scripts/analyze.py"
uv run --managed-python --project tests python "$experiment/scripts/paired_check.py"
uv run --managed-python --project tests python "$experiment/scripts/paired_analyze.py"
uv run --managed-python --project tests --with matplotlib==3.11.2 python "$experiment/scripts/plot.py"
```

Before preparation, compare downloaded SHA256s with `provenance/data-manifest.json`:
GloVe `544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4`,
SIFT `dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984`.
Preparation records these hashes but does not enforce the historical values.
It recomputes exact subset neighbors with a direct float64 C oracle; published
full-corpus neighbors are unsuitable for subsets. Full downloads total about 1 GB.
Building 24 graphs and timing both phases takes sustained CPU time.

The original `report.py` is retained only in `recorded/scripts/`: its prose embeds
this run's conclusions. Review new measurements before writing a new report.
Historical plots and summaries remain unchanged. Timing intervals describe one
machine and seed; they do not establish production or cross-platform performance.
