# Past experiment summaries and reproduction

This archive retains written reproduction instructions, protocols and result
summaries. Past scripts, lockfiles, raw receipts, figures, traces, environments,
downloaded inputs, databases and compiled outputs have been removed from the
working tree. Current application examples, tests and reusable benchmark tools
remain outside this historical archive.

## Reproduce a past experiment

The last complete pre-cleanup source snapshot is
`4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd`. Git history is preserved; no history
rewrite or permanent object purge is part of this cleanup.

1. Create an isolated checkout of that snapshot:

   ```sh
   git clone --no-checkout https://github.com/cercova-studios/sqlite-vec.git /tmp/sqlite-vec-experiment-replay
   git -C /tmp/sqlite-vec-experiment-replay checkout --detach 4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd
   cd /tmp/sqlite-vec-experiment-replay
   ```

2. Open the experiment's README/report listed below in that checkout. Follow its
   exact toolchain, input acquisition, seed, baseline/candidate revision and
   feature-profile instructions. Native builds need the documented SQLite vendor
   files; `bash scripts/vendor.sh` obtains those explicitly. Install dependencies
   in an isolated environment as the report describes.
3. Run that snapshot's replay commands with fresh database/output paths. Restore
   any prerequisite experiment inputs in the order specified by the report.
   Use its independent oracle, failure controls and hash checks before interpreting
   new measurements.

The pre-cleanup snapshot restores harnesses and committed receipts; it is not a
claim that every measurement used that revision of the native engine. Use each
report's original source revisions and hashes. Ignored downloads, databases,
models and build directories must be regenerated; uncommitted raw outputs cannot
be restored from Git. Reproduction depends on the stated upstream assets and
toolchains remaining available. Fresh timings and other nondeterministic results
can differ. The summaries below are historical findings, not fresh test passes.

## Results by experiment

| Experiment | Recorded result and interpretation | Reproduction/details |
| --- | --- | --- |
| HNSW alpha pruning | The synthetic alpha=1.1 throughput claim did not generalize across GloVe/SIFT and paired timing. Keep alpha=1.0. | [2026-10-26-hnsw-alpha](2026-10-26-hnsw-alpha/README.md), [report](2026-10-26-hnsw-alpha/REPORT.md) |
| Metadata compression | Real stored-blob/codec probes round-tripped, but showed no SQL/query or production-size benefit. Defer general compression; narrow packing remains a hypothesis. | [2026-10-26-metadata-compression](2026-10-26-metadata-compression/README.md), [report](2026-10-26-metadata-compression/REPORT.md) |
| SciFact hybrid retrieval | Full-corpus nDCG@10 was vector 0.7127, FTS 0.6695, hybrid 0.7130. Hybrid-minus-vector uncertainty included zero; defaults stayed unchanged. | [2026-10-26-scifact-hybrid](2026-10-26-scifact-hybrid/README.md), [report](2026-10-26-scifact-hybrid/REPORT.md) |
| Scoped hybrid retrieval | Eligibility/lifecycle checks passed on controlled coordinates. Larger fusion budgets could reduce relevance; this did not establish real-model quality or a universal hybrid gain. | [2026-10-26-scoped-hybrid](2026-10-26-scoped-hybrid/README.md), [report](2026-10-26-scoped-hybrid/REPORT.md) |
| Filtered FTS | Correlated EXISTS preserved all measured tuples/metrics while reducing full/scoped keyword p50 from 1255.75/752.21 ms to 4.00/2.78 ms. Tiny-scope costs remained. | [2026-11-01-filtered-fts](2026-11-01-filtered-fts/README.md), [report](2026-11-01-filtered-fts/REPORT.md) |
| Adapter keyword SQL | Bundled-SQLite IN/EXISTS comparisons matched all 600 ordered results/BM25 doubles. Full/scoped p50 became 3.74/2.62 ms; EXISTS was adopted with small-scope tradeoffs. | [2026-11-02-adapter-fts](2026-11-02-adapter-fts/README.md), [report](2026-11-02-adapter-fts/REPORT.md) |
| Bounded witness indexing | Hash/list indexes and linear path output improved tested synthetic witness medians by 17.0–102.3 times while preserving path checks and budgets; index scratch increased some byte requirements. | [2026-11-03-bounded-witness](2026-11-03-bounded-witness/README.md) |
| Comparison follow-up | SQL IN materialization dominated a measured filtered case; sparse flat reads amplified copied bytes. Graph and engine controls motivated bounded-witness and sparse-read work, not a database migration. | [2026-11-03-comparison-followup](2026-11-03-comparison-followup/README.md), [report](2026-11-03-comparison-followup/REPORT.md) |
| Engine comparison | Fork and Zvec were competitive on small filtered workloads; Zvec was about 2.7 times faster on one SIFT 10%-eligible case. No engine won all shapes or justified a universal ranking. | [2026-11-03-engine-comparison](2026-11-03-engine-comparison/README.md), [report](2026-11-03-engine-comparison/REPORT.md) |
| Adaptive sparse flat reads | Held-out random-0.1% medians improved SIFT 2.478→1.700 ms and SciFact 0.514→0.390 ms. Dense/small-vector fallback prevented larger regressions; some cases still regressed slightly. | [2026-11-03-sparse-flat](2026-11-03-sparse-flat/README.md) |
| Local datastore packaging | Isolated C/Rust consumers, included-input invalidation, per-connection registration and default/IVF local bundles were checked on macOS arm64. This was not cross-platform release qualification. | [2026-11-07-local-datastore](2026-11-07-local-datastore/README.md) |
| Capture/history and Rust review | Session plus explicit request/revision metadata was selected for a prototype. Raw inverse changesets rewind revisions. Later writer/index prototypes and Rust review counterexamples are distinguished from shipped write tools. | [2026-11-07-write-history](2026-11-07-write-history/README.md), [review](2026-11-07-write-history/rust-review.md), [prototype](2026-11-07-write-history/rust-prototype.md) |
| Lance DROP repair | Dataset retention and failed-DROP lifetime fixes passed 70 hybrid tests and native/sanitizer controls. Seven existing Clippy diagnostics and formatting gaps remained. | [2026-11-08-lance-drop](2026-11-08-lance-drop/README.md) |
| Late interaction | Frozen SciFact reranking reduced nDCG@10 from selected dense 0.708804 to RRF+late 0.634724. Both relevance gates failed; native MaxSim was rejected for that configuration. | [2026-11-08-late-interaction](2026-11-08-late-interaction/README.md), [report](2026-11-08-late-interaction/REPORT.md), [protocol](2026-11-08-late-interaction/SPEC.md) |
| Full writer/history and synchronization | Fifteen task-only measured cases passed an independent oracle. Separate source/FTS5/vec0 synchronization acceptance passed; indexed performance/public transport and the existing cancellation-test failure remained open. | [2026-11-08-writer-sync](2026-11-08-writer-sync/README.md), [acceptance](2026-11-08-writer-sync/ACCEPTANCE.md) |
| Lance identity/conversion | Final checks recorded 88 hybrid plus 66 native passes and four skips; actual conversion OOM controls returned SQLITE_NOMEM without staged writes. Seven existing Clippy findings remained. | [2026-11-10-lance-identity](2026-11-10-lance-identity/README.md) |
| Contributor onboarding | Two context-free agent trials completed native setup and priority recovery. Vendor fail-fast handling, host SIMD preservation and queue/provenance guidance were repaired; Python remained an optional wrapper over Make. | [2026-11-11-agent-onboarding](2026-11-11-agent-onboarding/README.md) |
| Dependency onboarding | A fresh agent built and relocated a separate consumer. Five installed-bundle checks passed; the retained live starter example fixed quoted backup paths and refuses disabled assertions. | [2026-11-11-dependency-onboarding](2026-11-11-dependency-onboarding/README.md) |

## Active-work exception

The in-progress BlitzRank experiment is excluded from completed-experiment
teardown. Its owner copied 17 required input/helper files (179,933,010 bytes) into
its own ignored workspace; their SHA256 hashes were independently verified before
old copies were removed. Its active directory, virtual environment, upstream
checkout and task cache remain intact. Current source examples/tests, shared
development Python environments and shared Cargo/uv/model caches are preserved.

## Cleanup verification — 11 November 2026

| Removed category | Files | Logical bytes |
| --- | ---: | ---: |
| Inactive onboarding/dependency scratch | 5,050 | 285,023,770 |
| Local Rust targets and experimental build variants | 27,229 | 6,853,488,605 |
| Completed experiment files and ignored inputs | 22,421 | 2,611,188,661 |
| Remaining old bundles, libraries and probes | 441 | 366,095,939 |
| Local bytecode and test caches | 71 | 2,416,924 |
| Total | 55,212 | 10,118,213,899 |

These are removed logical file sizes, not a measurement of net filesystem free
space. Git history, shared caches and active work remain. The archive removal
also unlinked three virtual-environment symlinks without touching their targets.
The retained archive has 39 historical Markdown records plus this index across
18 completed experiments. No completed-experiment process required termination.

Checks actually run after retirement: all ten bootstrap boundary tests and four
bundle manifest/failure tests passed; offline locked Cargo metadata resolves the
adapter's library and binary to existing sources. The retired `keyword-profile`
research example target was removed because its harness is now historical.
Affected-document local links and 87 pinned historical links were checked against
filesystem/Git objects. The active run's 17 preserved copy hashes were independently
verified before and after archive removal. Source examples, core tests and the
unrelated HNSW/.claude work are excluded from this cleanup.

An isolated checkout of the pinned snapshot was restored and removed after
complete HNSW-alpha and late-interaction offline receipt checks passed. The first
bare-Python HNSW attempt failed with `ModuleNotFoundError: No module named numpy`;
rerunning in the documented Python 3.12.13/NumPy 2.1.3 environment passed both
checks. Late interaction retained its recorded `all_gates_pass: false` result.
No live experimental measurements, native builds or full Rust/Lance/platform
suites were rerun during cleanup. Local generated libraries/bundles and Rust targets
were removed; rebuild them through the normal contributor/bundle guides before
using those tools. Raw ignored outputs are not backed up; use the reproduction
procedure above to generate new runs.
