# Filtered-vector comparison — 2026-11-03

The fork is competitive with Zvec on these small local workloads, but neither
wins every filter shape. Zvec is substantially faster on the SIFT 10%-eligible
case. These results support further targeted investigation rather than an engine
migration or a universal SQLite performance claim.

Measured on an Apple M5 Pro (18 cores, 48 GiB RAM), macOS arm64.
The comparison covers upstream sqlite-vec 0.1.9, the committed fork at `8614154`
(excluding the unrelated dirty HNSW helper), Zvec 0.8.0 and LanceDB 0.41.0.
The [methodology and replay](README.md) define the corpus hashes, SDK costs,
resource settings, filter semantics, query settings and measurement limits.

## Latency at observed mean recall ≥0.99

Median warm SDK call latency, milliseconds. Each entry selects the fastest tested
configuration meeting the threshold for that case. The raw summary retains the
configuration, actual recall, minimum query recall, p95 and all other settings.
This is matched observed mean recall, not a per-query recall guarantee.

| Dataset / eligible set | Upstream exact | Fork exact | Fork HNSW path | Zvec HNSW | LanceDB indexed, best tested layout |
|---|---:|---:|---:|---:|---:|
| SciFact / all | 1.038 | 1.017 | 0.103 | 0.103 | 1.242 |
| SciFact / random 10% | 1.094 | 1.082 | 0.087 | 0.443 | 1.423 |
| SciFact / random 1% | 1.076 | 0.903 | 0.015 | 0.058 | 1.184 |
| SciFact / coordinate band 1% | 1.107 | 0.873 | 0.015 | 0.058 | 1.156 |
| SIFT 50k / all | 3.262 | 3.097 | 0.098 | 0.097 | 1.397 |
| SIFT 50k / random 10% | 5.570 | 5.426 | 0.961 | 0.353 | 1.709 |
| SIFT 50k / random 1% | 3.693 | 3.801 | 0.070 | 0.069 | 1.438 |
| SIFT 50k / coordinate band 1% | 3.714 | 3.452 | 0.052 | 0.068 | 1.421 |

Unfiltered SIFT: fork/Zvec both achieve recall 0.999 at ef=128. Their p95 values
are 0.133/0.127 ms; the tiny median difference is not evidence of superiority.
At random 10% eligibility, both achieve recall 1.0 at ef=32 in this sample, with
p95 1.073/0.379 ms. Zvec's median is about 2.7 times faster there. The cause of
that gap was not profiled, so no HNSW algorithm change is proposed.

The fork can use exact ranking over selective eligible-ID sets. Its HNSW-path
numbers must not be described as pure ANN graph traversal. Zvec likewise has
adaptive query strategies. Empty controls pass and are retained separately.

LanceDB tests one and 64 partitions with unquantized IVF_HNSW_FLAT. One partition
helps unfiltered latency: SIFT 1.397 ms versus 2.564 ms with 64 partitions, and
SciFact 1.242 versus 1.702 ms. At SIFT random 1%, the 64-partition layout is faster
(1.438 versus 1.507 ms). Neither layout is claimed optimal. Exact LanceDB medians
are 1.961 ms on unfiltered SciFact and 4.292 ms on unfiltered SIFT; its full results
remain in the summary. These small warm SDK calls do not test LanceDB's larger
columnar/out-of-core workload strengths.

## Setup and resource costs on SIFT 50k

| Path | Setup + ingest + index/flush seconds | Whole-process peak MiB | Post-build database MiB |
|---|---:|---:|---:|
| Upstream exact | 0.24 | 71.3 | 27.4 |
| Fork exact | 0.18 | 71.5 | 27.4 |
| Fork HNSW | 5.96 | 107.4 | 45.3 |
| Zvec HNSW | 5.79 | 174.0 | 38.2 |
| LanceDB exact | 0.40 | 312.9 | 24.8 |
| LanceDB indexed, 64 partitions | 2.84 | 457.2 | 56.3 |
| LanceDB indexed, 1 partition | 7.66 | 405.5 | 57.4 |

RSS includes Python, inputs, imports, SDK objects, indexing and queries; it does
not isolate native index memory. Storage includes retained versions and engine
metadata. The fork stores more bytes than Zvec here, despite lower process RSS.
Build timings include engine import/startup. Upstream uses a wheel with unknown
compiler flags; the fork uses the recorded Apple M1/NEON build. Their difference
cannot be attributed solely to source changes.

## Correctness, failures and limits

All exact paths achieve tie-aware recall 1.0. Every live quality and timed call
checks ID uniqueness, eligibility, count bounds and reported distance units against
original vectors. Missing ANN results reduce recall instead of being discarded.
All workers check fresh handles after reopening in-process. This is not crash
recovery or transaction/rollback testing across these engines.

The offline verifier regenerates strict/tie-aware recall, mean-target selection
and summaries from returned IDs and frozen masks/oracles. Tests include a cutoff-
tie counterexample where returning extra tied neighbors must not compensate for
missing strictly closer neighbors. Corrupted IDs and recall values are rejected.

A LanceDB scalar-index setup probe stalls with one I/O thread at both one and two
compute threads (15-second bounds). Two I/O threads complete both probes. The
initial real worker was stopped after roughly 2.5 minutes with only 0.45 seconds
of CPU. Final runs use one compute worker and two Lance I/O threads; the bounded
probe and logs are retained. No native stack diagnosis or external bug report was
made. Timings affected by an earlier per-query distance-column warning were
rejected; final projections explicitly request the distance column.

This is one build/insertion order per engine/dataset and 100 queries per dataset,
with three warm rounds. No CPU affinity, independent-host isolation, confidence
intervals, cold-cache, multi-client, update/delete costs, durability equivalence,
Turso/libSQL, LadybugDB, zvec-grep or SQLite–Lance comparison is included.
No production defaults or code changed, and no push or CI was added.

The next useful diagnostic is the fork's SIFT 10%-eligible gap against Zvec,
followed by a larger corpus and more insertion orders. Hybrid relevance and graph
composition should use their own workload contracts rather than reuse this table
as evidence of an overall database ranking.
