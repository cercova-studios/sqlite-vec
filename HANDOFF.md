# Handoff — read this first

## Scoped hybrid retrieval evaluation — completed controlled slice 2026-10-26

The metadata compression archive was committed first in `52d1f14`.
The requested synthetic retrieval evaluation is now complete and archived in
[benchmarks-ann/experiments/2026-10-26-scoped-hybrid/README.md](benchmarks-ann/experiments/2026-10-26-scoped-hybrid/README.md).
Production-corpus relevance and downstream agent usefulness remain unverified.
No engine/default change, push or CI. Preserve the unrelated HNSW helper edit
and .claude/ worktrees. The concurrent search-lab commit `d8800a2` changes no
measured C source/header hashes and remains outside this evaluation.

The evaluation compares vector-only, FTS-only and the existing published RRF
recipe using sixteen authored procedure records, explicit relevance labels and
nine queries. Distractors grow from 0 to 1,000/10,000; three insertion orders,
branch budgets 10/50 and nine timed rounds cover baseline, committed invalidation,
atomic scope/text/vector replacement and reopen. Flat four-dimensional authored
vectors isolate recipe behavior; they are not learned production embeddings.
A separate eligibility model and exhaustive distance/fusion checks validate
1,944 quality points and 17,496 timed queries. No ineligible or expired/invalidated
results; reopened IDs equal post-update IDs in every configuration.

Baseline macro recall@10 at zero distractors: hybrid 96.4%, vector 82.1%,
FTS 61.9%. At 10,000 distractors: hybrid/vector 53.6%, FTS 0% on adversarial
keyword matches. At 1,000 distractors, increasing hybrid budget 10 -> 50 lowers
recall 53.6% -> 39.3%. At 10,000/budget 10, pooled median query latency is
0.66 ms vector, 16.4 ms FTS and 17.0 ms hybrid. These descriptive synthetic
measurements do not justify new ranking defaults. The unanswerable query still
returns unrelated vector/hybrid results: nonempty output does not imply usefulness.
Late global filtering loses all deployment candidates at both nonzero sizes.

Retained raw receipts include result IDs, judgments, wall/CPU timings, insert,
invalidate/update and FTS-optimize costs, allocated/free pages and final file
bytes. Source/library/script/fixture/guide hashes and query plans are archived.
The measured C build uses committed sources at `52d1f14`, excluding the existing
HNSW edit. Six evaluation tests pass, including pending writes, savepoint/full
rollback, commit/reopen and four deliberate unsafe-filter mutations. Three
existing vec_rrf loadable tests pass. Archive verification regenerates metrics;
full application suites/sanitizers were not rerun for this benchmark-only work.

Recommended next step: a representative labelled corpus and real embeddings,
with a specified read/write mix. Retain these eligibility/lifecycle checks and
compare all three baselines. Investigate the filtered FTS query path with plans
and profiling before proposing a schema or engine optimization. Compression,
TEXT pruning, broader buffering and Lance version cleanup remain separate scope
decisions. Keep alpha=1.0; push and CI remain deferred.

## HNSW alpha experiment — completed 2026-10-26

User requested realistic-dataset recall/QPS comparisons against the current
build, several ef_search settings, experimental only, no push or CI.
Plan: temporary isolated builds at alpha 1.0 (unchanged baseline), 1.05,
1.1 and 1.2, scaling the diversity rejection distance in hnsw_select.
Result: keep alpha=1 unchanged. The full sweep completed 24 builds, 96 quality
points and 480,000 timed queries. Three insertion orders per dataset/variant,
five rounds per ef. A separate seed-0 alternating alpha-1/1.1 check completed
80 pairs (160,000 timed queries) and reproduced per-query recall and neighbor
counts. Graph/libraries use isolated snapshots; production source hashes match
pre-experiment snapshots, including the preserved unrelated helper edit.

Alpha=1.1 initial GloVe mean gains were 7.7–13.9% across ef settings, with
high-ef recall losses. Alternating timing reduced high-ef gains to +3.65% at
128 (seed-0 recall -0.57 pp) and +5.66% at 256 (-1.03 pp). Low-ef timing
intervals include no gain. SIFT has no consistent timing improvement. The
historical synthetic +22–45% gain did not reproduce. Alpha=1.05 is a candidate
for a target-corpus follow-up, not a validated new default.

User decision: keep alpha=1.0. The durable archive is
[benchmarks-ann/experiments/2026-10-26-hnsw-alpha/README.md](benchmarks-ann/experiments/2026-10-26-hnsw-alpha/README.md),
with the [report](benchmarks-ann/experiments/2026-10-26-hnsw-alpha/REPORT.md),
scripts, compressed raw receipts, summaries, figures and provenance. Original
scripts are preserved verbatim; runnable copies use checkout-relative paths.
Downloaded datasets and compiled libraries are excluded. This archive and
HANDOFF are committed locally at the user's request; no production change, push
or CI. Temporary downloads may be purged; no production-corpus, million-scale,
cold-cache, concurrent-client, quantized-vector, high-dimensional modern-embedding
or cross-platform validation. Timing CIs cover ten paired rounds on seed 0 only.

M=16, ef_construction=200; ef_search=32/64/128/256; k=10. Real ANN-Benchmarks
GloVe-100 (cosine) and SIFT-128 (L2), deterministic 50k train subsets and 1k
held-out queries, exact neighbors recomputed for each subset. Three matched
insertion orders and repeated timing rounds; warm sequential single-client
queries. Never use published full-corpus truth for the subsets.

- [x] Download/validate public data and save hashes/provenance.
- [x] Build baseline/candidates from identical current source snapshots.
- [x] Validate correctness, exact oracle and reproducible graph seeds.
- [x] Run matched experiments, analyze recall/QPS variance, retain receipts.
- [x] Report recommendation and limitations; leave production alpha unchanged.

Oracle note: NumPy BLAS matmul produced non-finite output for finite normalized
inputs in this environment. No such output is used. Independent direct float64
C computes exact top-10 (plus the 11th distance for tie detection), validated
against elementwise NumPy for ten queries per dataset and exact SQL scans.
One SIFT query has a cutoff tie; report strict ID recall and tie-aware recall.
Baseline alpha-1 graph hash and answers equal current dist/vec0 on a 1k-row
smoke workload. Source snapshots include the preserved working-tree helper edit
identically in every variant; only the alpha comparison literal changes.

Artifacts: `benchmarks-ann/experiments/2026-10-26-hnsw-alpha/`. Production source and the
unrelated hnsw_allowed edit stay unchanged. Downloading/parsing binary data
only; no source instructions from external datasets are executed. h5py is
installed only in uv's temporary environment, not project dependencies.

Archive verification (2026-10-26): checksums and verbatim receipt copies pass;
both summaries regenerate exactly; all four runnable variant builds match the
recorded source hashes. The relocated baseline matches current-build graph and
query results, and both dataset exact-SQL/oracle checks pass. A corrupted-file
canary is rejected. Python syntax and relative document links pass. The current
loadable build passes with its existing DiskANN unused-parameter warning.
Full application suites and the timing sweep were not repeated for archiving.
The staged whitespace check reports only original Matplotlib SVG trailing
spaces and CSV CRLF endings; these generated artifacts remain byte-for-byte
intact to preserve their recorded hashes. No source/docs whitespace errors.

## Optional metadata compression assessment — completed 2026-10-26

User authorized assessment only. No production code, format or dependency
changes, new commits, push or CI. Recommendation: defer general compression;
consider a narrow opt-in integer-packing experiment only after measuring a
real metadata-heavy/read-mostly workload. Boolean metadata is already bit-packed;
whole-chunk codecs penalize existing eight-byte point reads and in-place updates.

Measured actual vec0 metadata blobs for ten synthetic distributions at chunk
sizes 128/1024/4096; 30 stored profiles and 60 zlib level-1/6 byte-exact
roundtrips verified. Level-1 plus assumed 16-byte framing at 1024 rows:
- categorical integers: 8192 -> 1210 bytes;
- random 64-bit integers: raw fallback (8219 encoded/framed > 8192);
- balanced Booleans: raw fallback (155 > already-packed 128 bytes).
Illustrative four-column schema saves 66.9% of metadata payload, but only
0.69% of vector+metadata payload at 768 float32 dimensions versus 13.43% at
32 dimensions. These are payload projections, not database-size or SQL-speed
claims. No production codec is selected; integer frame-of-reference sizes
are estimates only. Unknown constructor options currently fail at connect;
old numeric point readers lack codec-header checks, so any future compressed
format must be rejected by unsupported readers before data is accessed.

Durable archive (2026-10-26):
[assessment, scripts and raw receipts](benchmarks-ann/experiments/2026-10-26-metadata-compression/README.md).
Original files are preserved verbatim with SHA256s; a path-adapted runnable
benchmark writes fresh output separately. Compression remains unimplemented.
Archive verification checks hashes, Python syntax, all 30 profiles and 60 codec
receipts. A fresh replay passes SQLite equality and all 60 roundtrip assertions;
historical timing values are preserved, not claimed reproduced.
Measured macOS arm64, Python 3.12.13, SQLite 3.53.1, zlib 1.2.12. Nine timing
batches of 20 calls after five warmups. No production data, partially filled
chunks, end-to-end SQL performance/RSS or implementation fault tests. Shared
source notes were inaccessible. App suites were not rerun for this assessment.

## Approved-work closeout — 2026-10-26

All approved feature slices and the two defects found during review are now
committed locally, through `c85e736`. Fresh closeout runs pass all eight
mixed-index failure scenarios and all 34 native unit functions. No additional
defect or production change in this closeout. Prior full-suite/sanitizer
receipts remain in the repair entries below; they were not rerun here.

The user selected metadata-compression assessment, completed in the entry
above. Compression implementation remains unapproved. Compression, TEXT
zone maps and algorithm tuning stay optional; push and CI stay deferred.
Preserve the unrelated hnsw_allowed edit and .claude worktrees. The later
alpha archive request explicitly includes committing HANDOFF locally.

## HNSW failed-savepoint retry repair — approved 2026-10-26

Completed locally in `c85e736`; no push or CI changes.

User chose to fix code and preserve the documented retry guarantee. Scope:
protect pending edges during nested statement rollback inside a flush;
retain full-transaction rollback and normal ROLLBACK TO behavior. Cache structs
stay allocated through rollback; only their contents are reset. Protect all
HNSW columns while any is flushing, so another column's dirty lists survive.
No persisted-format or SQL-option changes, push or CI. Preserve the unrelated
hnsw_allowed helper edit and untracked worktrees.

Plan (single agent):
- [x] Add regressions: statement ABORT loses connectivity in either column
  at early/later writes; full ROLLBACK controls retain committed rows.
- [x] Protect nested rollback, verify retry, real savepoint rollback and reopen.
- [x] Run mixed-index failure matrix, full default/IVF/hybrid and sanitizers.
- [x] Update architecture and stage scoped hunks.
- [x] Commit locally.

SQLite can defer xSavepoint until the first subsequent write; the regression
exercises both paths and checks existing logical writes and failed row absence.

Verification: all eight new cases pass; four ABORT cases failed on the baseline
with wrong/missing HNSW neighbors, while four ROLLBACK controls passed. Both
columns and early/later edge writes are covered. The eight-case flat/IVF/HNSW/
DiskANN failure matrix now passes retry, independent distances, commit and
reopen checks; receipt /tmp/vec-hnsw-retry-matrix.json.

Full default suite: 493 passed / 141 skipped, experimental IVF: 630 passed /
4 skipped; 219 snapshots each. Hybrid rebuilt offline with the saved protoc.
ASan/UBSan: 61 focused tests pass (21 outside selection) on isolated source with
the unrelated helper change excluded; Python leak detection disabled and an
anchor connection prevents library unloading. No-HNSW scalar native build
passes 34 functions. Exact staged source passes all 38 HNSW tests and 34 native functions.
Default dist/vec0 restored. Existing unused DiskANN
parameter, hybrid unwind-table linker warning and NumPy stress warnings remain.
Sanitizer and candidate source copies use TemporaryDirectory and are removed.
No production datasets, other operating systems or explicit leak checks run.

## Integration review — HNSW savepoint flush failure, 2026-10-26

Review found a reproduced defect; the user approved repairing code to match
the documented guarantee. See the repair entry above. ARCHITECTURE.md says dirty lists
remain pending until an entire flush succeeds. Actual nested rollback clears
the cache during hnsw_flush, discarding pending edges while inserted nodes remain.

Minimal case on current default build: create v float[2] indexed by hnsw(m=4),
install a fault-injection trigger rejecting INSERT on t_hnsw_edges00 when
new.rowid=2, commit setup, insert rowids 1..30 with vectors [rowid,0]. Opening
SAVEPOINT s raises constraint failed and leaves the transaction active.
After dropping the trigger, query [20.1,0], k=3 returns only rowid 10, while
all 30 vectors remain. A failed COMMIT instead rolls back to zero rows as intended.

The four-index matrix uses flat, IVF, HNSW and DiskANN columns in one vtab.
Flat and IVF faults at read/savepoint/commit boundaries all pass retry,
independent L2 checks, commit and reopen (six scenarios). HNSW savepoint
failure loses connectivity: all 130 vectors remain but query 65 returns only
rowid 120, distance 55. The later HNSW commit matrix case was not reached;
the minimal single-index commit control above passes.

Source anchors: sqlite-vec.c vec0RollbackTo calls hnsw_cache_clear_all without
an active-flush guard; sqlite-vec-hnsw.c hnsw_cache_reset frees/zeros dirty
state during sqlite3_step in hnsw_edges_store; hnsw_flush still holds that cache.
Needs explicit active-flush ownership and correct full rollback handling, not
an unconditional exemption from rollback. Preserve the existing unrelated
hnsw_allowed helper edit.

Reproduction scripts: /tmp/vec-mixed-flush-review.py and
/tmp/vec-mixed-flush-matrix.py. The matrix stopped at its first failure;
its final JSON receipt is therefore absent. No new commit, push or CI changes.

## DiskANN visited-set repair — 2026-10-26

Completed locally in `bc9f22e`; no push or CI changes.

User continued after the reproduced stall diagnosis. Scope: grow visited sets,
handle zero rowids, propagate visited-growth allocation failure, preserve graph
format and search/prune settings. No push or CI changes; preserve unrelated
HNSW edits and untracked worktrees.

Plan (single agent):
- [x] Observe the SQL regression time out at 15 seconds on the old build and
  the rowid-zero unit regression fail before changing production code.
  The independent allocation regression also fails on an isolated baseline
  (inserted < 0), after bypassing the earlier zero-case failure.
- [x] Grow the set at half capacity with checked doubling and malloc64, retain
  members if growth fails, track zero separately, mark before expanding edges.
- [x] Verify growth/OOM/retry/signed-rowid units and original workloads;
  full default suite and rebuilt hybrid, sanitizer and portable unit builds.
- [x] Complete experimental-IVF integration and exact candidate checks.
- [x] Commit the four scoped files locally.

Final verification: experimental-IVF full suite 622 passed / 4 skipped and
219 snapshots. Exact candidate with committed HNSW passes 34 native functions
and all six SQL regression cases. Default dist/vec0 restored.

Current evidence: default full suite 485 passed / 141 skipped, 219 snapshots;
ASan/UBSan units 34 functions pass; scalar no-HNSW/no-IVF/no-rescore units pass.
Six SQL regressions cover beams 8/128, first rowids 1/0/-500, 1,000 inserts,
unique/sorted query results with independent L2 distances, savepoint rollback,
commit and reopen. They use child deadlines so a C loop cannot hang pytest.

Fault injection in a temporary instrumented source copy (committed HNSW) also
passes under ASan/UBSan: force only the visited-set resize allocation to fail;
insert 51 returns SQLITE_NOMEM, row and full-vector counts stay 50, retry
succeeds, and a query allocation failure can be followed by a successful query.
Diagnostic scripts /tmp/vec-diskann-oom-{driver.c,check.py,result.log} retain
that recipe. Instrumented copies are removed; no fault flags enter production.
The normal unit OOM test uses SQLite's real hard heap limit and restores it.

Memory now scales with unique visits; resizing temporarily retains old and new
arrays. No fixed visit cap or algorithm tuning was added. No production-corpus
or other-OS measurements; query cancellation remains outside this repair.

## DiskANN stall investigation — 2026-10-26

User authorized investigation, keeping commits local and CI deferred.
Root cause reproduced; no production fix or new commit in this investigation.

A fresh -O3/-g default build at `dist/diskann-investigation/vec0.dylib`
reproduces a hang with deterministic Gaussian float[8] vectors (seed 42;
the script first generates eight 8-d centers, then generates the inputs).
For binary quantization, default n_neighbors=72:
- search_list_size=8: 82 inserts finish; insert 83 exceeds a 3-second timeout.
- search_list_size=128: 727 inserts finish; insert 728 exceeds that timeout.
- Identical vectors, two repeated values, eight repeated centers and positive
  random controls each finish 1,000 inserts with both search-list sizes.
The recorded duplicate-only diagnosis was too narrow; mixed-sign random
vectors reproduce the underlying stall. Timing is diagnostic, not a speed claim.

`diskann_search` initializes `DiskannVisitedSet` to 4 * searchListSize,
rounded to a power of two. `diskann_visited_set_insert` silently returns zero
when full, just as for an already-present rowid. Search ignores that return.
After an untracked candidate gets dropped, approximate-distance insertion can
reintroduce it as unvisited. The beam size does not bound unique graph visits.
Temporary instrumentation stopped after 10,000 steps with visited/capacity
32/32 (beam 8) or 512/512 (beam 128). Changing only initial visited capacity
to 128 * searchListSize in a temporary build makes both 1,000-row cases finish.
That is a causal diagnostic, not a durable fix: any fixed multiplier can fill.
The instrumented candidate used committed HNSW, excluding its unrelated edit.

Recommended next design: allow the visited set to grow with unique visits,
propagate allocation failure, and preserve the once-per-row expansion invariant.
Review rowid zero (currently the empty sentinel), capacity arithmetic, search
error cleanup and rollback before implementation. A hard visit limit would alter
ANN behavior and must not be substituted silently. No algorithm tuning was done.

Evidence/scripts (temporary files, not tracked):
- /tmp/sqlite-vec-diskann-repro.py and /tmp/sqlite-vec-diskann-repro.json
- /tmp/sqlite-vec-diskann-diagnose.py and /tmp/sqlite-vec-diskann-diagnose.json
- /tmp/sqlite-vec-diskann-boundary.json

All subprocesses had deadlines and were reaped. Temporary instrumented source
and library copies were removed. The main source, unrelated HNSW edit, default
dist/vec0, .claude worktrees and CI remain untouched. No production-corpus,
query-cancellation, sanitizer or cross-platform checks in this investigation.

## Unit harness repair — approved 2026-10-26

Completed locally in `ac7c60d`; no push or CI changes.

Scope: remove copied internal declarations, include the actual C core in the
unit translation unit, separate interleaved test functions, and use the current
IVF flag. Preserve assertions and test scenarios; change stale expectations only
against current source evidence. Production behavior stays unchanged.

Plan (single agent):
- [x] Reproduce the existing compile failure and inspect source/history.
- [x] Restore independent rescore, IVF and DiskANN tests and remove header drift.
- [x] Run native, sanitizer and relevant feature-disabled builds; check all cases
  are called and assertions cannot be disabled silently.
- [x] Update current build documentation, review explicit paths, commit locally.

No push or CI changes. Preserve the unrelated dirty sqlite-vec-hnsw.c edit,
.claude/ and HANDOFF (untracked during that repair).

Verification this session:
- Original `make test-unit` failed compilation with duplicate enum/struct types
  and nested function definitions. The repaired default NEON build runs all 33
  original test functions, each called once; optimized scalar also passes.
- ASan/UBSan NEON (`-O1`, `halt_on_error=1`) passes all 33 functions.
- IVF off: 30 functions pass; rescore/HNSW off with IVF on: 29 pass;
  all four optional indexes off: 8 pass. Full-scan rollback stays outside guards.
- NDEBUG compile canary fails with "Unit tests require assertions" as intended.
- An isolated candidate with committed HNSW passes all 33 functions; changing
  rescore's zero-sign predicate from >= to > makes the existing bit test fail.
- Preserved all original parser scenarios; unknown-index tests now use
  unknown_index because hnsw is supported. Rescore int8 expectations now match
  the current fixed [-1,1] mapping/clamping, retaining the original inputs.
- Removed the private copied header and its only include. Remaining old names
  are historical notes and untouched .claude worktree copies.
- No runtime sources changed. Python/hybrid integration suites, other operating
  systems, fuzzing and production datasets were not rerun for this harness-only
  repair. No new hooks or CI enforcement were installed.

## Flat-vector batching — approved 2026-10-26

Completed locally in `e3076e7`; no push or CI changes. Unrelated HNSW edits
and .claude/ worktrees remain untouched.

Approved: coalesce consecutive flat-vector blob writes in up to 64 KiB per
vector column. Keep rowids, validity and metadata immediate. Flush before
reads, commands, savepoints, commit, deletes, noncontiguous writes and chunk
switches; never in xRelease. Rollback discards pending ranges. No storage
format or SQL option changes; compare with current HEAD 9f73413, commit locally,
no push or CI. Preserve the unrelated HNSW helper edit.

Plan (single agent):
- [x] Build unchanged baseline and demonstrate deferred-write regression.
- [x] Add bounded contiguous buffers; cover transaction/read/command/resource
  paths and preserve blob-size validation. Keep rescore/IVF/DiskANN untouched.
- [x] Verify element types, updates/deletes, partitions, mixed indexes/storage,
  pressure, rollback/savepoints, failures, reopen and interruptions.
- [x] Benchmark against the baseline, update architecture and usage docs,
  verify default/IVF/no-HNSW/hybrid/sanitized builds, and commit explicit paths.

Final validation: default 479 passed / 141 skipped; IVF 616 passed / 4 skipped;
219 snapshots in each full run, including all rebuilt hybrid tests. ASan/UBSan
and exact staged sources each pass 104 focused tests (4 unrelated skips) and
150 snapshots. Sanitizers use halt_on_error, with leak detection disabled for
Python and an anchor connection retaining the instrumented library. No-HNSW
passes 41 flat/metadata/partition/zone-map tests and 150 snapshots; one mixed
HNSW/rescore case is explicitly deselected. Default dist/vec0 was restored.
Unit target and production datasets were not run. No push or CI changes.

The deferred-write test first failed on the baseline because vector bytes were
already persisted. New tests cover float/int8/bit inserts, float UPDATE,
deletes/reuse and complete chunk removal, nested/statement/full rollback,
rename/reopen, partitions, two connections, buffer-cap boundaries, mixed
rescore/HNSW/IVF columns, query_only-induced flush failure at read/commit/
savepoint, retry, randomized SQL-oracle results and interruption at each read
step. The baseline rejects int8/bit UPDATEs after SQLite loses their subtype;
the tests preserve that existing rejection rather than expanding this patch.

Snapshot helpers and the smoke/delete shadow-storage assertions now issue
a virtual-table read before inspecting persisted vector bytes. All expected
values and all snapshot fixtures remain unchanged. The old per-vector blob
write helper was removed after replacing both callers; blob-size validation
and its evidence identifier remain. Slot bounds precede offset arithmetic.

Final synthetic receipts compare a fresh baseline built at HEAD 9f73413
(plus the preserved unrelated HNSW helper edit) with the candidate in separate
processes. Each workload uses deterministic Gaussian float32 vectors and ten
independent exact-KNN/distance checks; results match across builds.

| Dimensions | Rows | Baseline write s | Batched write s | Shadow changes (both) | WAL bytes (both) |
|---|---:|---:|---:|---:|---:|
| 32 | 10,000 | 0.01832 | 0.01483 | 30,020 | 1,573,872 |
| 128 | 10,000 | 0.03083 | 0.02078 | 30,020 | 5,529,072 |
| 768 | 3,000 | 0.14348 | 0.03025 | 9,006 | 9,595,512 |

These single-run consecutive-insert results do not establish production
speedups. Query timing is not a speed claim. Frequent reads, partitions and
noncontiguous writes reduce batching; rowid/validity/metadata writes remain
immediate and WAL traffic was unchanged. Receipts are
/tmp/sqlite-vec-flat-batch-{32,128,768}-final.json. Temporary test/build dirs
were scoped with TemporaryDirectory and removed. Build variants remain ignored.
TEXT zone maps, compression and optional whole-chunk/metadata buffering remain
outside the completed flat-vector slice. CI stays deferred.

## IVF batching — approved 2026-10-26

Completed locally in `9f73413`; no push or CI changes. The unrelated HNSW
helper edit and untracked .claude/ worktrees remain untouched.

Approved design: buffer up to 64 packed IVF cells and 8 MiB per table. Flush
before reads, commands, savepoint opening, rename and xSync, or under pressure.
Keep rowid mappings and quantized columns' original-vector KV writes immediate.
Clear pending cells on rollback; never flush in xRelease. Reuse unused validity
bits after deletion instead of treating live-row count as the next slot.
No persisted format, quantizer, centroid-selection, API, push or CI changes.

Plan (single agent):
- [x] Build unchanged current IVF baseline; add failing deferred-write and
  deleted-slot regressions.
- [x] Buffer cell mutation and batch whole-cell writes; wire all read/command,
  savepoint/rollback, rename and lifetime paths. Remove obsolete statement fields.
- [x] Verify quantizers, trained/untrained queries, filters, deletions, failed
  statements/flushes, nested savepoints, buffer pressure, connections and reopen.
- [x] Compare time, shadow changes, WAL and independent query results against
  the baseline; update architecture and benchmark docs; verify build variants
  and commit explicit paths locally.

Review focus: flush failure retains pending state; savepoint rollback removes
only later writes; pressure does not drop cells; reused slots preserve live
vectors; centroid commands cannot read stale cells. Reads may incur a flush.

Validation: final full experimental-IVF suite 604 passed / 4 skipped;
default 467 passed / 141 skipped; 219 snapshots in each. Rebuilt hybrid tests
ran in both full suites. Exact staged C sources (using the committed HNSW
helper instead of its unrelated dirty edit) built and passed 99 IVF tests.
No-HNSW + IVF build: 107 IVF/zone-map tests passed; the mixed HNSW/IVF case
was explicitly deselected because it needs HNSW. Default dist/vec0 restored.

ASan/UBSan: 99 IVF tests passed with halt_on_error enabled; leak detection
was disabled for the Python host. An initial repeated-load invocation aborted
inside load_extension; the successful runs kept an anchor connection open
so the instrumented library stayed loaded. No sanitizer findings were reported.
All temporary build/test directories used TemporaryDirectory and were removed.
Unit target and real-corpus workloads were not run. No push or CI changes.

Regressions first failed on the baseline for immediate cell writes and for
overwriting the last live row after deleting a middle slot. The new suite
covers all seven quantizers, trained/untrained cells, point and KNN reads,
rowid filters, nested/savepoint/statement/full rollback, ID reuse, rename,
reopen, multiple columns/connections, mixed IVF/HNSW writes, both buffer
limits, partial flush failure at read/commit/savepoint, retry, randomized
mutations with an independent oracle, and interruption at each read step.
Existing centroid-command interruption tests remain unchanged and pass.

Ruling: flush owns a detached buffer while nested SQLite statements run.
Nested UPDATE statement rollback must retain the pending logical edits; full
transaction rollback advances a generation to discard them. This prevents
use-after-free and loss of earlier vectors during a partial read flush.
SQLite savepoint callback errors report a generic constraint code rather
than the trigger message. Existing column readers can map nested interrupts
to SQL logic error (also reproduced on the baseline at step 162); the new
interruption check accepts those existing error forms but checks every live
vector's value after recovery. Old shadow-count helpers now issue a vtab read
before inspecting flushed counts, with their expected values unchanged.

Benchmark baseline was freshly built before edits from current HEAD d45f575
(plus the preserved, behavior-equivalent HNSW helper edit) as
dist/ivf-batch-base/vec0.dylib. Each case inserts 10,000 float[32] Gaussian
vectors after 512 seed rows; trained cases probe all 16 centroids. Both builds
run in separate processes. Final single-run results:

| Case | Baseline write s | Batched write s | Shadow changes | WAL bytes (both) |
|---|---:|---:|---|---:|
| Untrained | 0.08781 | 0.02613 | 50,167 -> 40,324 | 1,771,632 |
| Trained | 0.07828 | 0.04266 | 50,168 -> 40,382 | 1,878,752 |
| Trained int8 | 0.08226 | 0.04873 | 60,167 -> 50,385 | 2,521,472 |

Results match across builds; each case gets recall@10 1.00 against the
independent float32-vector oracle and checks original-vector distances.
These timings do not establish production speedups or reduced WAL traffic.
Receipts: /tmp/sqlite-vec-ivf-batch-{untrained,trained,int8}-final.json.
Flat-chunk batching, TEXT zone maps and compression remain open.

## HNSW write batching — approved 2026-10-26

Completed locally in `d45f575`; nothing pushed and no CI changes.

The user approved HNSW neighbor-write batching first. IVF and flat-chunk
batching remain open. Keep node/vector writes immediate and reuse the HNSW
cache for read-your-writes; mark changed neighbor lists dirty and flush each
once in xSync, before opening savepoints and before cache eviction. Rollback
clears cached state after SQLite restores shadow tables. No format, SQL API,
dependency, push or CI changes. Preserve the unrelated hnsw_allowed edit.

Implementation plan (single agent; approved design above):

- [x] Add a regression that sees pending graph changes through KNN while
  shadow edges remain unchanged until commit; confirm failure on the baseline.
- [x] Add dirty flags and hnsw_flush_all(vec0_vtab *), wire xSync/savepoints
  and cache-reset paths. Preserve rebuild and cache_mb=0 behavior.
- [x] Exercise inserts, deletes, updates, nested savepoints, statement failure,
  failed commit, cache pressure, rename, multiple columns/connections and reopen.
- [x] Benchmark shadow-row changes, transaction time and retained WAL bytes
  against the baseline. Update current architecture docs, run default,
  experimental-IVF, no-HNSW and hybrid checks, then commit explicit hunks locally.

Review focus: failed flush must abort without losing older dirty state;
rollback after a boundary flush must restore the right graph; cache eviction
must flush before reset; deleted/reused IDs must not resurrect edges; Lance
provider reads and maintenance must share the same transaction boundary.

Ruling: do not flush in xRelease. SQLite releases B-tree savepoints before
calling it (vendor/sqlite3.c, OP_Savepoint / sqlite3VtabSavepoint). Injecting a
neighbor-write failure there left partial writes and no named savepoint to
roll back. The regression test fails on that implementation. Keeping lists
pending across release preserves rollback; the next xSavepoint or xSync flushes.

Verified final source: 467 passed / 109 skipped on default; 572 passed / 4
skipped with experimental IVF; 219 snapshots in each. The rebuilt hybrid passes
all 30 HNSW tests; all 21 hybrid tests also ran in the final full suites.
The no-HNSW build passes 26 metadata/zone-map tests and 140 snapshots. The
exact staged C sources, excluding the unrelated hnsw_allowed edit, also build
and pass all 30 HNSW tests. Default dist/vec0 was restored and its flags checked.
Unit target and real-corpus workloads were not run. No push or CI changes.

Tests include 100 deterministic rounds of four random mutations with an
independent nearest-neighbor oracle, nested/statement rollback, full rollback,
reused IDs, multiple columns, cache pressure, cache_mb=0, rename/reopen,
cross-connection visibility, rebuild with/without cache, denied xSync writes
and a trigger-injected partial flush failure. The pre-feature skip-write test
failed as intended (49 prematurely written edges); the xRelease regression
also failed before removing that unsafe flush.

Final synthetic benchmark: 3,000 Gaussian float[32] vectors, seed 17, m=16,
ef_construction=100, ef_search=100, cache_mb=256, one transaction. Compared
with the retained prior default build (dist/zonemap-default/vec0.dylib):
shadow-row changes 48,693 -> 15,219; insert+commit time 0.1208 -> 0.1057 s;
retained WAL 1,145,392 -> 1,091,832 bytes. Committed graph hashes and query
results match across builds; pending queries match after reopening. These
single-run synthetic timings do not establish production performance. With
cache_mb=0 both builds have 48,693 changes and 1,145,392 WAL bytes.
Receipts: /tmp/sqlite-vec-hnsw-batch-final.json and
/tmp/sqlite-vec-hnsw-batch-uncached.json. Temporary test/build directories were
scoped with TemporaryDirectory and removed; build variants remain ignored.

IVF/flat-chunk batching and compression remain open. Frequent statement
savepoints or cache eviction reduce batching; flush scans resident cache flags.
The on-disk graph format and neighbor-selection policy remain unchanged.

## Scoped hybrid recipe adoption — 2026-10-25

The user explicitly requested coordination with "Process YouTube video"
(`01a175bc-83e4-7a00-8d7b-a7c0dfe09414`) to adopt useful findings from the Anders
Swanson AI Engineer talk. The human request was verified with read_thread.
The proposed adoption brief and full synthetic probe were reviewed at:
`/Users/rohit/.codex/visualizations/2026/10/26/01a175bc-83e4-7a00-8d7b-a7c0dfe09414/aie-sqlite-adoption/`.
The recording itself was not reprocessed in this chat.

Adopted, separately from the zone-map commit: a scope/validity recipe in
local commit `9348893`. Validation: 3 focused RRF tests passed; full default
457 passed / 109 skipped and 219 snapshots passed. The coordinating chat's
independent review reported no actionable issue, using its own copied binary;
our integration checks used the rebuilt current library. No push or CI changes.

The recipe is in
site/guides/hybrid-search.md and test_vec_rrf_scoped_hybrid_recipe in
tests/test-loadable.py. Eligibility is applied inside both vector and FTS
candidate branches before their limits, using the existing explicit rowid IN
constraint. The test executes the actual published SQL, checks half-open validity
boundaries, excludes wrong-tenant/invalidated/expired/future records, preserves
branch ranks and source IDs, and tests immediate invalidation/savepoint rollback
and a vector-only result. Applying the same policy only after both global top-2
branches gives no eligible rows; filtering before their limits returns IDs 5,6.

No engine, planner, fusion-function, retention or authorization behavior changed.
Keep RRF; the talk's illustrative weighted scores do not establish a better
calibration. Application redaction/lifecycle/access policy remains separate from
these retrieval predicates, and invalidation never deletes Lance recovery data.

Controlled synthetic evaluation is complete; see the latest entry above.
The remaining production-corpus follow-up compares vector-only, FTS-only and
hybrid queries under scope/time/status skew, invalidation, updates, growing
distractors and corpus size. Use independently specified expected records; measure ineligible
and stale results, relevant recall, empty-result rate, latency, write/maintenance
cost and retained bytes. Measure downstream agent outcomes separately only when
a real workload exists. Synthetic SQL correctness and the current ANN/zone-map
timings do not establish memory usefulness. No new CI, automatic feedback policy,
memory engine, real-corpus download or compression is included.

## Approved zone-map pickup — 2026-10-25

The owner approved numeric/Boolean metadata zone maps with conservative persisted
bounds and fallback when statistics are absent. TEXT and compressed metadata are
deferred. Partition predicates already filter chunks in vec0_chunks_iter.
Completed and committed locally as `902a333`; nothing was pushed and no CI changed.

Implementation plan (single agent):

- [x] Add a failing SQL test that corrupts an excluded chunk's metadata blob;
  pruning must avoid reading it, while missing stats must use normal filtering.
- [x] Store native 16-byte min/max blobs in the existing _info table, one
  zonemap_<column>_<chunk> key per numeric/Boolean column in each new chunk.
  Seed empty entries on chunk creation, widen after metadata writes, remove
  entries on whole-chunk deletion. Never recreate missing entries for a partial
  existing chunk. Use SQLite transactions rather than a data cache.
- [x] Prune strictly excluded chunks before metadata reads in the flat KNN loop.
  Mirror existing integer/float/Boolean comparison semantics; preserve full i64
  precision. Keep missing or malformed entries on the normal scan path.
- [x] Verify comparisons, boundaries, IN filters, inserts, updates, deletes,
  failed statements, savepoints, rollback, old chunks, rename and reopen.
  Benchmark filtering with and without stats and report write/storage cost.
- [x] Update architecture/current usage documentation, run default, experimental
  IVF and no-HNSW checks, then commit explicit paths locally. Do not push or add CI.

Limits: deleting extrema leaves wider bounds and reduces pruning effectiveness.
Existing chunks remain unindexed without backfill. Older extension writers do
not maintain these entries; clear zonemap_* keys before downgrade writes.
The unrelated sqlite-vec-hnsw.c edit and .claude worktrees remain untouched.

Validation: the new skip test failed on the previous build with "Could not
filter metadata fields" and passes with pruning. Final checks: 456 passed / 109
skipped on default; 561 passed / 4 skipped with experimental IVF; 219 snapshots
passed in each complete run. The rebuilt hybrid passes all 21 hybrid tests and
all 9 zone-map tests. A no-HNSW build passes 26 metadata/zone-map tests and 140
snapshots. The default dist/vec0 build was restored. Unit target not run; the
previous handoff reports its pre-existing duplicate-enum failure.

The zone-map suite checks 100 deterministic randomized insert/update/delete
steps against ordinary SQL, plus boundaries, full i64 precision, integer IN,
missing/malformed entries, failed statements, rollback, savepoints, two
connections, whole-chunk deletion/reuse, rename and reopen. No schemas, SQL
names, index search paths, snapshot fixtures, or CI workflows changed.

Final ordered-data benchmark (`benchmarks-ann/zonemap.py`, separate processes for
old/new libraries, 65,536 rows x 32 dimensions, 64 chunks, 64 warm queries):

- Candidate p50: 0.1000 ms with bounds vs 0.1729 ms with bounds removed (1.73x).
- Candidate p95: 0.1076 ms vs 0.1926 ms.
- Inserts: 0.1736 s candidate vs 0.1560 s previous build (about 11.3% slower).
- Stats: 64 entries, 1,024 payload bytes; keys/B-tree/page overhead excluded.
- Every query result is asserted. This favorable ordered case does not establish
  speedups for random metadata distributions or production workloads.

Raw JSON: `/tmp/sqlite-vec-zonemap-final.json`. The benchmark baseline uses
`git show f011be9:sqlite-vec.c`, compiled against the checkout's unchanged feature
includes and SQLite headers. The unrelated dirty HNSW helper edit was present
for both builds; it does not participate in these flat queries.

## Pickup update — 2026-10-25

The uncommitted Lance maintenance patch is finished for `optimize`. Both
`lance0` and `vec0 storage=lance` accept `insert into t(t) values ('optimize')`.
It compacts files and updates the rowid index during `xSync`; savepoint,
statement and transaction rollbacks restore the request state. The provider ABI
is version 2, with a version check before the C core reads `xCommand`.

The owner chose to finish optimize first. The unfinished `cleanup` command and
its direct `chrono` dependency were removed: deleting versions before SQLite
commits can destroy the version required after a failed commit. Old versions
remain on disk. Safe version deletion needs its own design.

Validation in this pickup: hybrid release build succeeded; the complete
`tests/test-*.py` run passed with 445 passed, 109 skipped and 219 snapshots passed.
The new tests cover all three storage modes, failed statements, savepoint and
transaction rollback, failed-commit recovery, compaction and old-manifest
retention. A separate no-HNSW build loaded successfully and registered graph_bfs.
Experimental IVF and the known-broken unit target were not run in this pickup.

On this macOS checkout, replacing an already-loaded hybrid dylib in place was
followed by exit 137 on load, despite a valid on-disk signature. Copying Cargo's
output to a new file and renaming it over `dist/vec0_lance.dylib` restored loading.
Do not rebuild/copy the library while tests are using it. The release linker
also warned that its `__eh_frame` exceeds the compact-unwind encoding limit.

The unrelated HNSW change to reuse `vec0_rowids_contains` was preserved.
Maintenance was committed locally as `96f6db8`; nothing was pushed. CI remains
deferred. The vector-index pickup below completes ordered queue item 1.

## Previous handoff

### Approved vector-index design and implementation plan — 2026-10-25

Maintenance was committed locally as `96f6db8`; the unrelated HNSW edit remains
outside that commit. The owner approved opt-in ANN activation through `optimize`.
The completed vector-index feature and benchmark are committed as `f011be9`.

Goal: `optimize` builds IVF_HNSW_SQ on the `lance0.vector` column at 16,384 rows.
Smaller tables and tables that have never opted in retain exact scans. The vec0
storage provider does not build a redundant Lance ANN index. Keep all dataset
maintenance in LanceStore::sync and record its final version inside SQLite.
Subsequent writes remain searchable through Lance's unindexed-row scan and the
existing pending-row merge; another optimize updates the vector index.

Use 8-bit SQ and Lance 13's default HNSW build parameters. Partition count is
rows / 256, clamped to 1..256. Search starts with 32 partitions, allows adaptive
expansion for selective filters, reranks 4*k candidates with original vectors,
and uses ef >= max(128, 4*k). These quality settings are benchmark candidates,
not a general accuracy guarantee. IVF_FLAT stores full vectors; IVF_PQ compresses
further but adds subvector training constraints. IVF_HNSW_SQ avoids those
dimension restrictions while keeping the index smaller than float storage.

Implementation (single agent, existing Rust/SQLite/Python tools):

- [x] Add a failing SQL regression in tests/test-hybrid.py: no ANN files before
  optimize, files after it, accurate L2 distances, prefilters, pending and
  committed inserts/updates/deletes, savepoint/transaction rollback and reopen.
- [x] Add the lance0-only indexed-column capability to LanceStore; create or
  refresh its vector index after compaction in xSync. Preserve old versions.
- [x] Add an assert-based benchmark in benchmarks-ann/lance.py, with exact NumPy
  neighbors, before/after SQL timing, recall, build time, disk and peak RSS.
  Run Gaussian and clustered vectors; target recall@10 >= 0.95 in both.
- [x] Update ARCHITECTURE.md and lance/README.md to describe opt-in approximate
  search and its costs. Run affected tests and the full default suite, inspect
  the final diff, then commit the feature separately. Do not push or add CI.

Review focus: query semantics after opt-in; exact reported distances; searchable
unindexed changes; failed-commit recovery; training and memory cost. Benchmark
evidence is local and synthetic, not proof of production recall or speed.

Results on this macOS checkout, 20,000 rows, 128 dimensions, 100 queries, seed 42:

| Dataset | Exact p50 / p95 (ms) | ANN p50 / p95 (ms) | Recall@10 | Optimize (s) | Peak process RSS (MiB) |
|---|---|---|---|---|---|
| Gaussian | 1.519 / 1.798 | 1.653 / 1.944 | 1.00 | 0.312 | 182.25 |
| Clustered | 1.665 / 2.117 | 1.427 / 1.996 | 1.00 | 0.312 | 186.34 |

Both datasets used 78 IVF partitions. The inspection example confirmed
IVF_HNSW_SQ and ANNSubIndex on vector_idx in the committed dataset version.
Retained dataset bytes grew from 10,349,078 to 16,374,949 (Gaussian), and from
10,349,075 to 16,415,213 (clustered). This includes old versions, not just the
live index. Gaussian queries were slower after indexing, so no universal speed
claim is supported. Raw JSON is in `/tmp/sqlite-vec-lance-{gaussian,clustered}-final.json`.

Validation: hybrid release build passed; 21 hybrid tests passed; the full
default run passed with 447 passed, 109 skipped and 219 snapshots passed.
Both final benchmark runs passed their recall and distance assertions. Small
tables, vec0 providers, odd vector dimensions, prefilters, unindexed changes,
rollback, failed commits, reopen and empty indexed tables are covered. No real
embedding corpus, experimental-IVF build or unit target was run for this feature.
Historical scope: the handoff stayed untracked during that feature commit.
The later alpha archive request includes HANDOFF in its local commit. The HNSW edit, .claude worktrees, push and CI are outside this pickup.

**For:** Codex (or any agent) picking up this tree cold.
**Status as of:** 2026-10-24, branch `ann-hnsw`, HEAD `53627db`, working tree clean except untracked `.claude/` (session worktrees — do not commit it).
**Do not confuse with upstream:** this is a fork of `sqlite-vec` becoming an "all-encompassing" SQLite extension. Everything below past `411e100` is fork work, not upstream.

## Where things stand

All planned tracks are done, merged, and green. Nothing is pushed.

| Commit | What |
|---|---|
| `411e100` | Hybrid Lance build + `vec0` storage-provider ABI (base for both worktrees below) |
| `ad429cb` | Lance column patches: `UPDATE` writes only changed columns (0.1 MB vs 14.5 MB on the bench in the commit message) |
| `4b6cdf8` | FTS5 + `vec_rrf` hybrid search docs + tests (DuckDB track) |
| `3385130` | One shared filtered-KNN rowid path across flat/rescore/IVF/DiskANN/HNSW (DuckDB track) |
| `ab525bc` | `graph_bfs` / `graph_shortest_path` table functions (Ladybug track) |
| `305900d` | `ORDER BY distance LIMIT n [OFFSET m]` on `vec0` KNN queries; also fixed a bug where any `OFFSET` errored the query (DuckDB track) |
| `340baa7` | `vec_file_each` for `.fvecs`/`.bvecs`/`.ivecs` (DuckDB track) |
| `816867e` | Graph+vector end-to-end (k-hop as KNN filter, vector-seed expansion, >4096-neighborhood HNSW) — `tests/test-graph-vector.py` (Ladybug track) |
| `42dea66` | Lance auto-compaction in `lance/src/store.rs` |

Merge commits: `f51d608`, `8d22b16`, `c82debd`, `53627db`.

**Last verified results (post-merge, run in this checkout):** 441 passed / 109 skipped on the default build, 546 passed / 4 skipped on the experimental-IVF build, 69 passed on a focused hybrid/storage/graph/filtered/limit/file-reader run; 219 snapshots pass. Default build was restored afterwards (`dist/vec0.dylib` is default; `dist/vec0_lance` is built).

## Root architecture in one paragraph

One SQLite extension, two layers. The C core (`sqlite-vec.c`, `sqlite-vec-*.c`) is SQLite-only: `vec0` with flat/rescore/IVF/DiskANN/HNSW indexes, sparse vectors, `vec_rrf` hybrid fusion, and MMR rerank. `lance/` is a Rust crate whose `build.rs` compiles the C core in, producing a single `dist/vec0_lance` that registers everything plus `lance0` (a Lance-backed virtual table) and the Lance storage provider that `vec0` selects with `storage=lance[, storage_uri='...']`. Rust may depend on C, never the reverse. `make hybrid` needs `protoc` (see "How to build & test" for the invocation that actually works here).

Also read: `ARCHITECTURE.md` (the most load-bearing internal doc), `reference.yaml` (the source of truth for what each function does), and `lance/README.md`. Ignore `TODO.md` for pickup purposes — it is the old `ann` branch plan, not the current queue; the queue is below.

## The hard-won invariants (do not break these)

1. **`KNN_ROWID_IN` (`[` in the `idxStr`) is the *only* rowid-filter path.** Subquery filters like `rowid in (select node from graph_bfs(...))` go through it. Do not add a second kind, do not change its encoding. The shared helpers it uses are in `sqlite-vec.c`: `VEC0_KNN_EXACT_FILTER_MAX` (4096), `vec0_rowids_rank_exact(...)`, and `vec0_rowids_contains(rowidsIn, rowid)` where `rowidsIn == NULL` means all rows.
2. **Filters can return fewer than `k` rows.** On filtered-index traversals (IVF/DiskANN/HNSW) when the filter is selective, fewer-than-k is legitimate, not a bug. Tests now say this out loud (see `tests/test-filtered-knn.py`). Any caller — especially graph-flavoured code — must not assume `len(results) == k`.
3. **`xBestIndex` never returns unbounded rows.** If neither `k = n` nor `LIMIT` is given, the planner errors ("A LIMIT or 'k = ?' constraint is required…"); if *both* are given, it errors too. `LIMIT + OFFSET > 4096` errors at filter time for the same reason. That ceiling matches the existing `k` cap.
4. **SQLite tells providers the truth about rollbacks.** `vec0` sets `xSavepoint`/`xRelease` so SQLite calls `xRollbackTo` on statement-level rollbacks; the storage-provider ABI in `sqlite-vec.h.tmpl` extends this to `xSync`, `xCommit`, `xRollback`, `xSavepoint`, `xRelease`, `xRollbackTo`. Anything durable before SQLite's commit window opens a hole — the storage shadow table (`<name>_storage`) records exactly which Lance dataset version is committed, and the next writer restores Lance to that version first. Never advance Lance past it outside this protocol.
5. **Only one implementation of the Lance commit rules.** `lance/src/store.rs` (`LanceStore`) is shared by `lance0` and the vec0 storage provider. Do not fork the logic; patches, versions, bucketed upserts, and index maintenance live there and only there. Column patches (`UPDATE` of a subset) are grouped inside `LanceStore::sync()` and written via `MergeInsertBuilder` (whole-row upserts and `RewriteColumns` patches) — follow that shape for new write paths.
6. **The HNSW cache is transaction-aware.** The per-connection cache invalidates on `xRollback` / `xRollbackTo` and on pager data version (`SQLITE_FCNTL_DATA_VERSION`). The graph-module CSR cache borrows this. New caches must respect both hooks (see `sqlite-vec-hnsw.c` for the original pattern and `sqlite-vec-graph.c` for the copy).
7. **Do not gate optional code on `SQLITE_VEC_ENABLE_HNSW` unless the code genuinely lives inside that index.** A real bug happened here: the `graph_register(...)` call sat outside the guard while `#include "sqlite-vec-graph.c"` was inside it, so a no-HNSW build was missing the definition. The graph include now lives outside the `#if` in `sqlite-vec.c` (around line 5532). When you add a module, check both build variants.
8. **`sqlite-vec-graph.c` is `#include`d, not linked.** The change is deliberate: it keeps the amalgamation as one translation unit and matches the other `sqlite-vec-*.c` feature files. The Makefile `$(TARGET_LOADABLE)` prerequisites now use `$(wildcard sqlite-vec-*.c)` so graph edits trigger a rebuild. Adding a `sqlite-vec-*.c` file? No Makefile edit needed.
9. **Text PKs in filters resolve through `vec0_rowid_from_id`.** `vec0_rowids_in_collect()` in `sqlite-vec.c` handles text-PK tables and silently skips missing ids. If you add a new index kind, route it through that helper instead of reimplementing filter resolution.
10. **Register new SQL names in `tests/test-loadable.py`.** Its `FUNCTIONS`/`MODULES` lists feed `test_coverage`, which asserts every listed name has a test. Add new functions/modules there, add real tests in a per-file suite (pattern: `tests/test-graph.py`, `tests/test-vec-file.py`), and update `reference.yaml` in the same commit.

## Gotchas — build & test

- `uv run` fails in this sandbox with "could not create cache path …/.cache/uv". Always prefix test commands with `UV_CACHE_DIR=/tmp/uv-cache-muse`. Pytest cache noise about `.pytest_cache` is harmless.
- **`make loadable CFLAGS="-DSQLITE_VEC_EXPERIMENTAL_IVF_ENABLE=1"` does NOT rebuild from a clean default tree.** You must `rm -f dist/vec0.dylib` first, then make. Same when coming back to the default build. Otherwise the tests quietly run against the old binary and the skips mislead you.
- The loadable `test-*.py` suite is amalgamation-style: run it via `make test-loadable` or explicitly `UV_CACHE_DIR=... uv run --managed-python --project tests pytest -q -p no:cacheprovider tests/test-*.py`. Running a single file with `-k` is faster for iterating on one feature, but the cross-checks (coverage, snapshot) only fire in the full run.
- `make test-hybrid` additionally needs `protoc`. Point it at the saved binary: `PROTOC=/private/tmp/claude-501/-Users-rohit-sandbox-sqlite-vec/2db5337b-6d4f-49a3-bab8-8a4e560ddd84/scratchpad/protoc/bin/protoc make test-hybrid`. Skipped tests with that reason usually mean `dist/vec0_lance` is stale or missing; `make hybrid` rebuilds it.
- `make test-unit` now uses the actual C core and enables experimental IVF. The prior duplicate-header and interleaved-function failures are repaired; see the unit harness entry above for validation. Do not disable assertions with NDEBUG.

## Remaining work (promised vs. left-open — be careful which)

Open items fall into two buckets: things the original briefs ordered and haven't been touched, and findings from the recent work that need a decision before code. Do **not** invent new features; the ordered list below is the queue.

Current status (reconciled 2026-10-26):

1. **Lance vector indexes: complete** in `f011be9`, opt-in through optimize.
2. **Numeric/Boolean zone-map pruning: complete** in `902a333`.
   Compressed metadata and TEXT pruning remain optional, unapproved work.
3. **Batched maintenance: completed approved slices** — HNSW neighbor writes
   (`d45f575`), IVF cells (`9f73413`), flat-vector ranges (`e3076e7`).
   Whole-chunk or metadata buffering would require a separate scope decision.
4. **Unit harness: repaired** in `ac7c60d`, including all 33 original functions.
5. **CI: deferred by the user.** No push, tags or workflows without instruction.

DiskANN stall investigation and production repair are complete; see the repair
entry above for verification and commit status.
Compression and algorithm tuning remain optional. Preserve unrelated edits.

Findings to decide on, not to code speculatively:

- **HNSW α pruning: investigated, still experimental.** The historical synthetic alpha=1.1 result was intentionally reverted. The new GloVe-100/SIFT-128 sweep and alternating timing confirmation are complete (report above); they do not justify changing the default. Keep alpha=1. Alpha=1.05 may merit target-corpus follow-up. Production code remains unchanged; no runtime alpha option is added.
- **Worst-case-optimal joins, Cypher subset (Ladybug track item 4): rejected by design memo, not forgotten.** SQLite's row-store planner cannot express a WCOJ inside an extension without forking SQLite; a Cypher engine is 4–8+ weeks duplicating SQL the host already provides. The two shipped functions (`graph_bfs`, `graph_shortest_path`) cover the dominant graph-vector composition. The thing that could flip the Cypher call is a request for variable-length path predicates — that argues for one more table function, not a language.
- **DiskANN visited-set exhaustion can stall inserts.** Confirmed by the investigation above with deterministic mixed-sign random vectors. Duplicate-only controls did not stall. The fixed-size visited set silently filled; `bc9f22e` adds growth and error propagation. This finding is resolved.

## Optional cleanup (only if asked)

The two worktree branches still exist: `worktree-agent-ad2af631ab1036179` and `worktree-agent-a6c739ef0e781a358` under `.claude/worktrees/`. They are fully merged into `ann-hnsw` (verify with `git branch --contains HEAD` / `git cherry`). Delete with `git worktree remove` only if the user asks; both have an untracked `vendor` symlink inside that was never committed and must stay that way.

## Conventions to keep

- **Feature commits carry `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` plus a body that explains *why* and cites the numbers when there are numbers.** Match that shape for large changes (`git log -3` for the tone).
- **Explicit paths in `git add`.** No broad `git add -A`; `.claude/`, `dist/`, `vendor` symlinks, and build artifacts must never be committed (the latter two are gitignored except for the worktree symlinks, which only exist because of the sandbox).
- **No push, no tag, no `reset --hard` to publish anything** — commits are local until the user asks. Treat anything outside this workspace as read-only unless told otherwise.
