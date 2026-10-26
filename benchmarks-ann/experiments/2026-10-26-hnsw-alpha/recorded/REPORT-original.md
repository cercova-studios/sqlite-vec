# Experimental HNSW alpha pruning: real-dataset benchmark

2026-10-26. Production alpha remains unchanged; no push, CI, production source edit or project dependency change.

## Recommendation

Keep the current alpha=1 behavior. The historical synthetic claim of 22–45% more QPS for about 0.5 percentage points less recall did not reproduce as a general result. On GloVe, alpha=1.1 has modest paired timing gains at high ef_search but loses recall; SIFT shows no consistent timing improvement. Alpha=1.05 is a narrower follow-up candidate on GloVe, not a validated default: it was measured in the initial sweep but did not receive the separate alternating confirmation.

At ef_search=128, the initial three-order GloVe sweep measured alpha=1.05 at +8.2% QPS with a -0.01 pp mean recall difference. That is promising experimental evidence, not a production recommendation: the separate alpha=1.1 timing check showed that time-separated gains can shrink substantially under alternation.

## Scope and data

Both corpora are real public datasets distributed by [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks/blob/main/README.md?plain=1): GloVe-100 word embeddings (cosine) and SIFT-128 image descriptors (L2). They are established benchmark corpora, not modern high-dimensional production embeddings. Full source files were downloaded over HTTPS; SHA256 hashes, HDF5 attributes and original row selections are retained.

| Dataset | Source train/test rows | Indexed sample | Held-out queries | Metric |
|---|---:|---:|---:|---|
| GloVe-100 | 1,183,514 / 10,000 | 50,000 | 1,000 | cosine |
| SIFT-128 | 1,000,000 / 10,000 | 50,000 | 1,000 | L2 |

Seed 42 selects train/test subsets without replacement; train and test come from the separate source splits. Exact top-10 neighbors are recomputed over each selected train subset. Published full-corpus neighbor IDs are not used as subset truth. One SIFT query has a tie at the tenth-neighbor cutoff; no selected GloVe query does. Strict ID recall is the main result, with tie-aware recall also preserved in the CSV/JSON.

## Controlled comparison

- Baseline: current source at c85e736b2e7ef008fbfac0733b0de96ad7ed99ef, including the pre-existing unrelated hnsw_allowed working-tree edit identically in all variants. Baseline graph and answers match the current dist/vec0 library on a deterministic 1,000-row smoke workload.
- Variants: alpha=1.0, 1.05, 1.1, 1.2. Only the rejection comparison in hnsw_select changes: alpha * distance(candidate, kept_neighbor) < candidate_distance_to_base. Alpha=1.0 retains the exact original source expression.
- The multiplier acts on the engine's native selection distance: squared L2 for SIFT and cosine distance for GloVe. It is not a multiplier on square-rooted L2 distance.
- M=16, ef_construction=200, ef_search=32/64/128/256, k=10, cache_mb=256, unquantized float32 vectors. Alpha changes graph construction; it is not introduced as a runtime option.
- Three insertion orders (seeds 0/1/2), matched across variants. Node levels are deterministically drawn from rowid. Variant/dataset job order is randomized; all jobs run sequentially in separate processes.
- Initial sweep: 24 builds, 96 quality points, five query-timing rounds per point, 480,000 timed queries. Each round uses 1,000 queries in a shuffled order after 100 warmups. No benchmark workers run concurrently.
- Timings are warm, single-client SQL in an in-memory SQLite database, including Python sqlite3 execution/fetch overhead. QPS is queries / full batch wall time, not the reciprocal of median query latency.
- Initial QPS is the geometric mean across insertion orders of each order's median over five timing rounds. Error bars show the observed order range, not confidence intervals. Recall is the mean over the three orders; p95 latency is summarized separately.

![Initial recall/QPS sweep](/tmp/sqlite-vec-alpha-experiment/recall-qps.png)

## Initial sweep results

All QPS and recall deltas below compare with alpha=1.0 at the same dataset, ef_search and insertion order. Recall differences are percentage points (pp), not relative percentages.

### GloVe-100 / cosine

| alpha | ef_search | Recall@10 % | QPS | QPS delta | Recall delta pp | p95 us |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 77.340 | 28,746 | +0.0% | +0.000 | 46.1 |
| 1 | 64 | 85.700 | 16,488 | +0.0% | +0.000 | 79.3 |
| 1 | 128 | 92.347 | 9,009 | +0.0% | +0.000 | 138.3 |
| 1 | 256 | 96.763 | 4,769 | +0.0% | +0.000 | 272.8 |
| 1.05 | 32 | 78.017 | 30,932 | +7.6% | +0.677 | 42.3 |
| 1.05 | 64 | 86.183 | 17,574 | +6.6% | +0.483 | 72.0 |
| 1.05 | 128 | 92.337 | 9,747 | +8.2% | -0.010 | 130.2 |
| 1.05 | 256 | 96.567 | 5,235 | +9.8% | -0.197 | 242.3 |
| 1.1 | 32 | 78.457 | 31,479 | +9.5% | +1.117 | 44.3 |
| 1.1 | 64 | 85.797 | 17,750 | +7.7% | +0.097 | 74.0 |
| 1.1 | 128 | 91.803 | 10,264 | +13.9% | -0.543 | 126.2 |
| 1.1 | 256 | 95.987 | 5,274 | +10.6% | -0.777 | 233.7 |
| 1.2 | 32 | 77.217 | 27,821 | -3.2% | -0.123 | 49.0 |
| 1.2 | 64 | 84.950 | 16,454 | -0.2% | -0.750 | 76.5 |
| 1.2 | 128 | 90.823 | 9,090 | +0.9% | -1.523 | 135.9 |
| 1.2 | 256 | 95.167 | 4,966 | +4.1% | -1.597 | 249.8 |

### SIFT-128 / L2

| alpha | ef_search | Recall@10 % | QPS | QPS delta | Recall delta pp | p95 us |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 95.923 | 37,725 | +0.0% | +0.000 | 32.6 |
| 1 | 64 | 99.033 | 21,687 | +0.0% | +0.000 | 58.2 |
| 1 | 128 | 99.887 | 12,443 | +0.0% | +0.000 | 98.0 |
| 1 | 256 | 100.000 | 7,051 | +0.0% | +0.000 | 174.8 |
| 1.05 | 32 | 96.953 | 36,801 | -2.5% | +1.030 | 35.2 |
| 1.05 | 64 | 99.423 | 21,420 | -1.2% | +0.390 | 58.8 |
| 1.05 | 128 | 99.923 | 12,692 | +2.0% | +0.037 | 101.2 |
| 1.05 | 256 | 99.993 | 7,071 | +0.3% | -0.007 | 178.7 |
| 1.1 | 32 | 96.790 | 37,542 | -0.5% | +0.867 | 33.5 |
| 1.1 | 64 | 99.243 | 21,744 | +0.3% | +0.210 | 61.0 |
| 1.1 | 128 | 99.860 | 12,607 | +1.3% | -0.027 | 101.1 |
| 1.1 | 256 | 99.980 | 7,356 | +4.3% | -0.020 | 178.4 |
| 1.2 | 32 | 95.783 | 38,000 | +0.7% | -0.140 | 34.4 |
| 1.2 | 64 | 98.690 | 22,476 | +3.6% | -0.343 | 59.1 |
| 1.2 | 128 | 99.707 | 12,825 | +3.1% | -0.180 | 101.3 |
| 1.2 | 256 | 99.937 | 7,463 | +5.8% | -0.063 | 177.1 |

## Alternating timing confirmation for alpha=1.1

The initial sweep showed timing variation, so alpha=1.0 and 1.1 were rebuilt for insertion seed 0 in separate persistent worker processes. The workers alternated measured rounds, with identical query orders and warmups. Only one worker runs queries at a time; the other remains idle. Both workers reproduced their original per-query recall arrays and stored neighbor counts. There are ten paired rounds per ef_search, or 160,000 additional timed queries across both corpora.

The paired QPS delta is the geometric mean of the ten per-round candidate/baseline ratios. Its 95% bootstrap interval resamples paired log-ratios (10,000 samples, seed 42). This interval covers timing rounds on one fixed graph only; it does not cover other insertion orders, machines or datasets. Recall deltas in this table are also seed 0, unlike the three-order mean in the earlier tables.

| Dataset | ef_search | Paired QPS delta | Timing-only 95% interval | Recall delta pp |
|---|---:|---:|---:|---:|
| GloVe | 32 | +5.85% | -1.23% to +13.05% | +0.750 |
| GloVe | 64 | +0.82% | -2.41% to +3.89% | +0.230 |
| GloVe | 128 | +3.65% | +2.36% to +5.08% | -0.570 |
| GloVe | 256 | +5.66% | +4.01% to +7.59% | -1.030 |
| SIFT | 32 | -4.85% | -11.55% to +2.46% | +1.020 |
| SIFT | 64 | -2.61% | -4.59% to -0.60% | +0.360 |
| SIFT | 128 | -0.35% | -2.73% to +2.42% | -0.020 |
| SIFT | 256 | -0.90% | -2.28% to +0.59% | -0.010 |

![Alternating timing confirmation](/tmp/sqlite-vec-alpha-experiment/paired-confirmation.png)

For GloVe at ef_search 128/256, the timing intervals are positive, but recall is lower. At 32/64 the timing intervals include no gain. For SIFT, every paired point estimate is slower; the ef_search=64 timing interval is negative, with an accompanying recall improvement. This is an accuracy/latency tradeoff, not a universal optimization.

## Construction and resource costs

| Dataset | alpha | Median insert+commit seconds | Paired build-time delta across orders | Stored neighbor IDs (mean) | Maximum worker RSS MiB |
|---|---:|---:|---:|---:|---:|
| GloVe | 1 | 7.93 | +0.0% | 1,243,260 | 164.0 |
| GloVe | 1.05 | 8.46 | +7.2% | 1,266,429 | 153.8 |
| GloVe | 1.1 | 8.31 | +5.7% | 1,268,566 | 153.9 |
| GloVe | 1.2 | 8.65 | +12.7% | 1,262,814 | 154.7 |
| SIFT | 1 | 4.85 | +0.0% | 1,027,247 | 185.3 |
| SIFT | 1.05 | 4.99 | +4.6% | 1,197,457 | 189.0 |
| SIFT | 1.1 | 5.31 | +8.6% | 1,267,561 | 188.2 |
| SIFT | 1.2 | 5.69 | +15.3% | 1,288,751 | 188.6 |

Build time includes vector binding, SQL insertion and commit, not only neighbor selection. Stored neighbor IDs sum all graph levels, not just level zero. RSS includes the Python host, mapped train data, native SQLite/HNSW state and validation allocations; it is not index-only memory. The alternating check keeps two workers resident; combined system RSS was not measured.

## Correctness and provenance checks

- Every initial quality point returns ten unique valid rowids, sorted reported distances, and 10,000 returned distances matching an independent float64 calculation within rtol=2e-5 / atol=2e-5. The initial sweep checks 960,000 returned distances; the alternating builds repeat the quality checks.
- Exact ground truth comes from the separately compiled direct float64 reference in exact_oracle.c. It uses neither sqlite-vec kernels nor BLAS and retains stable rowid tie ordering. Ten oracle queries per corpus were checked against elementwise NumPy; exact SQL scans for ten queries per corpus reached tie-aware recall 1.0.
- NumPy BLAS matmul initially produced non-finite values from finite normalized inputs in this environment. That preparation attempt failed before writing usable ground truth and none of its output was used. The direct reference and elementwise validation replaced it.
- SHA256 hashes cover the full downloaded HDF5 files, selected arrays, source snapshots and variant libraries. The build manifest stores exact compiler commands, warnings and the preserved working-tree diff.

| Corpus | Source SHA256 |
|---|---|
| glove-100-angular | `544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4` |
| sift-128-euclidean | `dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984` |

Maximum observed reported-distance error in the initial quality checks: 1.5256983e-05. Initial worker CPU/wall fractions span 0.982–1.000; no full-system saturation profile was collected.

## Reproduce and inspect

All experiment artifacts live under `/tmp/sqlite-vec-alpha-experiment/` and may be purged by temporary-directory cleanup. Production C source, default dist/vec0 and CI were left untouched. No local commit was created for this experiment.

1. Download the two source HDF5 URLs listed in data/manifest.json into data/ (the download artifacts were renamed from .hdf5.part after validation).
2. Run build_variants.py from the repository to compile isolated libraries from the current source; the saved manifest identifies the exact snapshot used here.
3. Compile exact_oracle.c with `cc -shared -fPIC -O3 exact_oracle.c -o exact_oracle.dylib`.
4. Run prepare_data.py with the test Python environment plus temporary h5py.
5. Run benchmark.py validate, then benchmark.py sweep. Set VECLIB_MAXIMUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 for these Python commands.
6. Run analyze.py, then paired_check.py and paired_analyze.py. The bootstrap method is described above and its output is retained in paired_summary.json.
7. Run plot.py with temporary matplotlib. Plots are available as PNG, SVG and PDF.

- [Summary CSV](/tmp/sqlite-vec-alpha-experiment/summary.csv)
- [Summary JSON including order ranges](/tmp/sqlite-vec-alpha-experiment/summary.json)
- [Paired timing measurements](/tmp/sqlite-vec-alpha-experiment/paired_timing.json)
- [Paired timing summary](/tmp/sqlite-vec-alpha-experiment/paired_summary.json)
- [Baseline/oracle validation receipt](/tmp/sqlite-vec-alpha-experiment/validation.json)
- [Dataset hashes and selection metadata](/tmp/sqlite-vec-alpha-experiment/data/manifest.json)
- [Build manifest and exact flags](/tmp/sqlite-vec-alpha-experiment/builds/manifest.json)
- [Full sweep harness](/tmp/sqlite-vec-alpha-experiment/benchmark.py)
- [Independent exact reference](/tmp/sqlite-vec-alpha-experiment/exact_oracle.c)

## Limits and next decision

These are 50k-vector subsets on one macOS arm64 host, with Python 3.12.13 and NumPy 2.1.3, warm in-memory SQLite and a 256 MiB HNSW cache budget. Compilation targets apple-m1/NEON; the exact CPU model and installed RAM could not be read through the sandbox. There is no CPU affinity pinning, thermal/frequency control, concurrent-client load, cold-cache test, million-scale graph, modern 384–1536-dimensional embedding corpus, mutation workload or quantized-vector test. Bootstrap intervals do not remove those limits. Alpha=0.8 was not retested. Production test suites were not rerun because production code was unchanged; this is not a release validation.

If this experiment is continued, first test alpha=1.05 on a target production embedding corpus and repeat interleaved timings at a specified recall target. Use the measured accuracy/latency frontier, not fixed-ef QPS alone. Do not expose a default-changing tuning knob or promote alpha=1.1 from these results.
