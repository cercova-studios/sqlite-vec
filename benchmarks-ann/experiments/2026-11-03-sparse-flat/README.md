# Adaptive flat-vector BLOB reads

Flat KNN now reads contiguous runs of surviving vectors when the filtered chunk
is sparse enough. Whole-chunk reads remain the fallback. The change adds one
private reader in `sqlite-vec.c`; it keeps the existing chunk scan, filtering
bitmap, distance/ranking code, full-chunk scratch buffer and transaction path.
No index, cache, storage format, public option or dependency is added.

The reader falls back when more than one-eighth of chunk slots survive, or when
surviving vector bytes plus 64 KiB per read run reach the whole-chunk byte count.
This is a conservative private cost heuristic, not a SQLite page-size assumption.
Consecutive selected slots share one read. Results remain at their normal chunk
offsets, and distance calculation only visits selected bitmap slots.

## Measured result

The initial unconditional-selective prototype regressed dense filters and small
vectors. The adaptive policy was fixed after exploratory queries 0–29 and tested
on all 100 queries, including 70 previously unused queries. Results below use
only that held-out suffix; no threshold was changed after this confirmation.

| Workload | Whole median ms | Adaptive median ms | Whole / adaptive |
|---|---:|---:|---:|
| SIFT, random 0.1% | 2.478 | 1.700 | 1.46x |
| SciFact, random 0.1% | 0.514 | 0.390 | 1.32x |
| SIFT, random 1% | 3.706 | 3.763 | 0.98x |
| SciFact, random 1% | 0.880 | 0.849 | 1.04x |
| SIFT, random 50% | 11.360 | 11.232 | 1.01x |
| SciFact, random 50% | 1.713 | 1.579 | 1.08x |
| SIFT, no filter | 3.633 | 3.515 | 1.03x |
| SciFact, no filter | 1.205 | 1.059 | 1.14x |

The two-dimensional control uses SIFT's first two coordinates with the same masks
and insertion order. Adaptive medians differ from whole reads by -3.7% to +2.3%
across its cases; this does not establish exact latency equality. On the original
SIFT 1% case the adaptive path is about 1.5% slower, so this is not a universal
sparse-query speedup. Dense and small-vector fallbacks avoid the much larger
regressions of unconditional selective reads. All per-case/strategy summaries,
including 5%/10%, coordinate-band and empty cases, remain in the receipts.

Separate instrumented first-query probes count vector API bytes, excluding
metadata/rowid reads and distinguishing copied bytes from physical disk I/O:

| Query | Whole calls / bytes | Adaptive calls / bytes |
|---|---:|---:|
| SIFT, random 0.1% | 33 / 17,301,504 | 50 / 25,600 |
| SciFact, random 0.1% | 3 / 4,718,592 | 6 / 9,216 |
| SIFT, random 1% | 49 / 25,690,112 | 86 / 21,519,872 |
| SciFact, random 1% | 5 / 7,864,320 | 52 / 79,872 |

The conservative fallback deliberately leaves substantial read amplification in
SIFT's 1% case. The existing rowid membership scan and SQL IN construction also
remain. Any direct rowid-mapping fast path needs a separate scoped design that
preserves all predicates. Sparse tail chunks can also benefit: SciFact's final
chunk has only 63 valid vectors, and an unfiltered adaptive query copies those
vectors rather than the unused payload slots.

## Method and correctness

Baseline is committed `dc20c85`. Whole, always-selective and adaptive libraries
use the same committed C/header snapshot, replacing only `sqlite-vec.c` for the
candidate. The always-selective control removes only the policy block from the
candidate copy. Libraries compile with identical Apple clang 21 -O3/M1/NEON flags.
The unrelated dirty HNSW helper is excluded. The C copies and generated databases
stay under ignored `.work/`; production source and application dependencies are
not rewritten by the harness.

Public inputs and masks match the [first engine comparison](../2026-11-03-engine-comparison/README.md):
SciFact 5,183 BGE-small float32 vectors at 384 dimensions, SIFT 50,000 sampled
vectors at 128 dimensions, 100 fixed queries each, and one matched insertion
order. The extra two-dimensional projection is a control, not an additional real
embedding dataset. Eight cases are full, random 50%/10%/5%/1%/0.1%, coordinate-band
1%, and empty. SQL rowid-IN eligibility comes from an indexed ordinary scope table.

Five rounds reverse the three-strategy process order on alternating rounds.
Each process warms five queries per case, then measures 100 SDK calls per case.
This yields **36,000 timed calls**, 25,200 on the held-out suffix. The final
campaign runs serially, without concurrent test/build jobs or instrumentation.
Calls include SQL execution/result conversion, exclude input conversion,
independent validation and database/index construction. CPU affinity is not
pinned, and no confidence intervals or insertion/build variance are claimed.

Every call checks unique eligible IDs, result count, original-vector float64 L2
distances and tie-aware exact top-k against brute force. All ordered rows and
reported distances also match the whole-chunk baseline exactly across strategies
and rounds. New cutoff/closer/tie oracles are archived; matching original cases
are cross-checked against the earlier frozen oracle. Offline verification checks
eligibility, tie-aware recall, paired rows, counts, finite values and regenerated
summaries. It rejects corrupted IDs/query numbers. Original-vector distance checks
run live; the offline verifier does not reconstruct every distance without inputs.
Byte instrumentation runs separately, checks matching rows and records read sizes
for the first query of each case; its timings are not used.

Local tests cover numeric/text IDs, combined rowid/partition/metadata predicates,
float L2/L1/cosine, int8 and bit vectors, distance predicates, fewer-than-k/empty
results, two vector columns, buffered own writes, updates/deletes, savepoint/full
rollback, failed multi-row writes, commit/reopen and failed-flush retry. Malformed
BLOBs still fail length checks before either strategy. C unit tests verify copied
and untouched ranges, dense fallback, expired-handle failures and fresh-handle
retry, including first/last slots and adjacent runs.

Verified locally on the isolated candidate library: 517 default tests passed /
141 skipped, and 654 experimental-IVF tests passed / 4 skipped; each run passes
all 219 snapshots. All 76 adapter/CLI/MCP/G0 tests pass. The C unit harness and
AddressSanitizer/UndefinedBehaviorSanitizer unit execution pass. Existing DiskANN
unused-parameter and NumPy stress warnings remain. The default library is restored
after the IVF run. No full hybrid/Lance or other-platform suite was run.

## Replay

First regenerate and verify the [previous public input arrays](../2026-11-03-engine-comparison/README.md).
Requires macOS arm64, `cc`, generated `sqlite-vec.h`, vendor SQLite headers,
Python 3.12 and NumPy 2.5.4 in an isolated benchmark environment. From the repo root:

```sh
python3 benchmarks-ann/experiments/2026-11-03-sparse-flat/bench.py
python3 benchmarks-ann/experiments/2026-11-03-sparse-flat/reads.py
python3 benchmarks-ann/experiments/2026-11-03-sparse-flat/verify.py --write
python3 benchmarks-ann/experiments/2026-11-03-sparse-flat/verify.py
```

The benchmark overwrites its generated databases/libraries and recorded results;
use a separate checkout to preserve checked-in evidence. It validates prior input
hashes before running. `--queries` and `--rounds` support small development probes;
the archived final campaign uses the defaults 100 and 5. The standard-library
verifier needs no NumPy or native engine, but checks the current candidate source
hash and earlier frozen oracle. `--write` updates archive hashes after validation,
not before. Runtime/library/input/script digests are retained with results.

These are warm SDK calls on one macOS arm64 host, with small query samples and
one insertion order. Other vector dimensions/layouts, cold-cache, concurrent,
million-row and other-platform workloads remain unverified. The payload buffer
still holds one full chunk; no RSS reduction is claimed. Hybrid Lance performance
is outside this slice; its separate storage path is not changed. No push or CI.
