# Local SQLite datastore roadmap

Current work queue, updated 2026-11-07. This supersedes the old ANN branch plan
and the separate extensionless TODO. [HANDOFF](HANDOFF.md) records completed
slices and verification; [ARCHITECTURE](ARCHITECTURE.md) describes shipped behavior.

Build one reusable SQLite platform for personal projects in `~/sandbox` and
`~/code`: ordinary application data, agent coordination, search/retrieval and
optional versioned state. New strategies belong here when they preserve SQLite's
transaction model and earn their runtime/maintenance cost through measurements.
A Beads-like application is a proving workload, not the datastore's domain model.

## Execution order

P1's local bundle is verified. The P4/V1 contract and capture investigation are
complete, and the isolated Rust writer prototype is verified. Next: measure the full
writer/history path and integrate source/FTS/vector synchronization before exposing
write tools. Develop P2/P3 retrieval
interfaces on that foundation. Exercise P5 throughout. V2/V3 branch/merge work
follows a reviewed history/identity design; V4 remote sync is a later gate.
Existing read-only CLI/MCP behavior remains available while opt-in write tools
are designed and verified. No push, publishing or CI changes are authorized.

## Rust review follow-ups

[All-Rust review and counterexamples](benchmarks-ann/experiments/2026-11-07-write-history/rust-review.md).
These are existing defects/gaps, not changes made by the task writer prototype.

- [x] P1: retain Lance datasets across rolled-back/committed DROP, preserve
  active readers and reject DROP of pending writes without losing state.
  [Regression evidence](benchmarks-ann/experiments/2026-11-08-lance-drop/README.md).
- [x] P1: enforce lance0 rowid uniqueness and SQLite conflict policies; reject
  NULL TEXT/BLOB conversion buffers before constructing Rust slices on OOM.
  Preserve rollback when first enlisted inside nested savepoints.
  [Regression evidence](benchmarks-ann/experiments/2026-11-10-lance-identity/README.md).
- [x] P2: reject invalid UTF-8 TEXT without lossy replacement; preserve NULL and
  empty TEXT/BLOB values. Evidence is in the same Lance regression receipt.
- [ ] P2: qualify Lance metadata by SQLite schema; preserve collation semantics.
- [ ] P2: track all included C/header inputs in the legacy Rust harness/micro builds.
- [ ] P2: make the adapter cancellation test deterministic without weakening its
  asserted execution boundary. Its observed Rust suite result is 9/10, not a pass.
- [ ] Repair the stale generated-version Rust binding lock and existing lint/format
  gaps; qualify fresh Lance builds once protoc is available. Do not bulk-rewrite code.

## P1 — Reliable local consumption (verified on macOS arm64)

- [x] Stage a standalone amalgamation for the Rust binding and regenerate it
  when an included C feature changes. Verify a consumer outside this checkout.
- [x] Repair native static/CLI feature dependencies and ship the static header.
  Reproduce stale-build failures, then verify clean and incremental consumers.
  [Verification receipt](benchmarks-ann/experiments/2026-11-07-local-datastore/README.md).
- [x] Resolve the host-linked Rust registration failure with per-connection
  registration; verify Apple, Homebrew and bundled SQLite independently.
- [x] Provide one project-local bundle/install path for the Rust CLI/MCP binary,
  matching native library, C amalgamation/header and a build manifest. Record
  source revision/hashes, SQLite version, platform and feature flags. Keep build
  profiles separate so an IVF/default switch cannot silently reuse an old binary.
- [x] Verify a clean project can install, inspect capabilities, write/read its
  SQLite data, back up, restore and reopen without requiring this source tree.
  Pin this fork explicitly; upstream registry packages do not contain local work.
  [Build and use a local bundle](docs/local-bundle.md). Other-platform validation
  remains open; these checks do not establish release readiness.

## P2 — Plain-text semantic and hybrid search

- [ ] Accept natural-language queries through a host-configured embedding path;
  agents should not have to construct float arrays. Preserve explicit-vector
  and embedding-reference calls, JSON/JSONL envelopes and read-only controls.
- [ ] Record model/revision, dimensions, metric, normalization and chunking
  identity for ingestion and queries. Reject incompatible embedding spaces;
  changing models must not silently mix indexes.
- [ ] Keep model execution outside the SQLite-only C core. Start with one
  supported local path and verify missing-model, cancellation and retry behavior.
- [ ] Measure end-to-end relevance and latency against current lexical/dense/RRF
  baselines, including empty results, scope and provenance.

## P3 — Expose existing search strategies

- [ ] Support trusted exact/HNSW/IVF/rescore configurations through the adapter;
  it currently admits only canonical flat/L2 tables and performs exact scans.
- [ ] Replace fixture-scale admission limits only after target-workload tests
  establish suitable work, memory and latency limits. Do not simply remove caps.
- [ ] Preserve eligibility before candidate limits, score units, missing values,
  approximation/completeness declarations, bounded output and stable failures.
- [ ] Profile shared HNSW metadata eligibility before changing graph traversal;
  preserve the single rowid-IN boundary and rollback-aware caches.
- [ ] Benchmark candidate budgets, exact reranking and selectivity-aware search
  against held-out queries and multiple insertion orders before adopting defaults.

## P4 — Ingestion and safe agent writes

- [x] Design a reusable write boundary over ordinary relational tables, with
  validation, stable IDs, revision checks, idempotency keys and atomic batches.
  Keep schema-specific workflows in applications, not a mandatory generic JSON store.
  [P4/V1 contract](benchmarks-ann/experiments/2026-11-07-write-history/write-contract.md).
- [x] Implement the fresh-fixture Rust transaction prototype: per-request Session
  capture, explicit request/revision ledger, two-agent claim, retry and safe undo.
  Keep public write tools disabled until its acceptance checks pass.
  [Prototype and tested limits](examples/write-prototype/README.md). It is isolated
  on rusqlite 0.40.2 because 0.41's Session build dependency cannot currently resolve.
- [ ] Add explicit opt-in write access beside the existing read-only adapter;
  preserve read-only callers, JSON contracts and host-controlled project scope.
- [x] Prototype atomic source/embedding/FTS5/flat-vec0 synchronization for fresh
  indexed fixtures. Prepared vectors stay outside the write lock; Session captures
  only canonical ordinary data. Undo rebuilds projections from saved vectors.
  [Indexed contract](examples/write-prototype/README.md#indexed-source-fixtures).
- [x] Prove atomic claim/update and stale-revision rejection with concurrent
  agents; test failed statements, savepoints, full rollback, cancellation and
  retried requests. SQLite's per-database single-writer constraint still applies.
  Verified for fresh task-only/indexed fixtures; Lance/provider writes remain unverified.
- [ ] Define schema versions and test backup/restore and crash recovery before
  using these writes on important project data. Existing-data migrations require
  a concrete migration/rollback review; initial implementations use fresh fixtures.

## P5 — Reusable proving workloads

- [ ] Integrate one real project and a small Beads-like task/dependency workflow:
  create records, find ready work, race two claims, update, search and fetch evidence.
- [ ] Freeze corpus/query revisions and compare relevance, ANN recall, p50/p95
  latency, memory, build/update cost and agent task success. Distinguish exact
  reported distances from candidate recall and semantic relevance.
- [ ] Exercise warm/cold reads, skewed/correlated filters, insert/delete churn,
  concurrent readers/writers, interruption and rollback; preserve counterexamples.
- [ ] Compare strategies against one SQLite exact oracle. Other engines remain
  optional isolated benchmark references, not application runtime requirements.
- [ ] Retain executable experiments, outcomes and negative results in the
  development archive; keep user-facing docs about current supported behavior.

## Dolt-inspired versioning track

### V1 — History, attribution and selective undo

- [x] Write the versioned-table contract: opt-in ordinary tables with stable
  primary keys, revision identity, actor/request metadata and transaction boundaries.
- [x] Evaluate SQLite Session/changesets against an explicit change log using
  measured capture, undo and recovery cases. Session needs host-build support;
  it does not capture virtual tables and is not a complete version-control engine.
  [Experiment and limits](benchmarks-ann/experiments/2026-11-07-write-history/README.md).
- [ ] Verify the complete Session+ledger/history-index pipeline before selecting
  a shipped capture default. Raw inversion rewinds revisions; undo needs a new revision.
- [ ] Keep canonical versioned data in ordinary tables. Treat FTS/ANN structures
  as derived, tied to a data/model revision; never merge index shadow-table blobs.
- [ ] Implement selective undo as a new validated change. Reject conflicts with
  intervening writes rather than overwriting another agent's work.
- [ ] Verify attribution and history are atomic with application writes, including
  crash/retry boundaries and all managed writer connections.

### V2 — Isolated workspaces and historical reads

- [ ] Design commit ancestry, branch heads, schema-version identity and isolated
  working states while retaining SQLite as the storage engine.
- [ ] Start with correct snapshots/changesets and measure copying, replay and
  storage costs. Efficient structural sharing is a distinct optimization target,
  not something file copies provide for free.
- [ ] Ensure checkout/historical queries use matching derived indexes or an
  explicitly correct fallback. Stale indexes must never silently serve another revision.

### V3 — Three-way merge and explicit conflicts

- [ ] Compare base/ours/theirs by stable row/field identity. Specify updates,
  deletes, unique/FK constraints, schema changes and missing common ancestors.
- [ ] Return structured conflicts and apply resolutions transactionally. Validate
  application invariants such as exclusive task claims and acyclic dependencies;
  merging disjoint cells alone does not establish semantic correctness.
- [ ] Use independent merge oracles, convergence/idempotency cases, adversarial
  branch histories, interrupted merges and rollback/recovery checks.

### V4 — Optional cross-machine synchronization

- [ ] Define revision/changeset exchange, deduplication, compatibility and conflict
  rules only after local versioning works. Changeset capture alone is not sync.
- [ ] Review transport/authentication separately before enabling networking.
  No hidden daemon, remote service or new engine is required for local workflows.

## Research candidates — evaluate before implementation

- [ ] ACORN-style filtered traversal after shared metadata profiling identifies
  residual graph-search cost; evaluate correlated predicates at matched recall.
- [ ] Audit the existing experimental RaBitQ/int4/FP16/BF16 implementations;
  compare extended/multi-bit RaBitQ with reference implementations and exact reranking.
- [ ] Study packed SIMD scoring/Faiss FastScan where IVF scans dominate. PQ4 and
  scalar int4 are different formats; do not assume interchangeable kernels.
- [ ] Evaluate chunking, hybrid weighting/budgets, graph expansion and optional
  host-side query–passage reranking on real project tasks. TurboQuant remains a
  research comparison, not a promised speedup or a requirement for local use.
- [ ] Study incremental graph repair under insert/delete churn if quality decays;
  keep alpha=1.0 unless a separate recall/QPS decision changes it.

Primary references: [SQLite Session](https://www.sqlite.org/sessionintro.html),
[SQLite isolation](https://www.sqlite.org/isolation.html),
[Dolt commit graph](https://www.dolthub.com/docs/architecture/storage-engine/commit-graph/),
[Dolt Prolly Trees](https://www.dolthub.com/docs/architecture/storage-engine/prolly-tree/),
[Beads' Dolt rationale](https://github.com/gastownhall/beads/blob/main/docs/architecture/dolt.md),
[ACORN](https://arxiv.org/abs/2403.04871),
[extended RaBitQ](https://arxiv.org/abs/2409.09913),
[Faiss FastScan](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-%28FastScan%29).
These are design leads; they do not establish capabilities of this fork.

## Untriaged low-level backlog

Preserve these candidates from the old `TODO` until current-code inspection
confirms whether each is still needed: partition updates/IN filters; NULL metadata;
BLOB/date/datetime support; UUID/ULID performance; NOT NULL constraints for
auxiliary/metadata/partition columns; dictionary encoding; vtab no-change handling;
auxiliary INSERT caching and batched auxiliary reads. They are not approved work
or production defects merely because the historical list mentioned them.

The [old ANN plan](benchmarks-ann/experiments/2026-11-07-local-datastore/legacy-ann-todo.md)
and [old core notes](benchmarks-ann/experiments/2026-11-07-local-datastore/legacy-core-todo.md)
are preserved as history, including unresolved historical checkboxes. They are
not instructions to publish, force-push, revive Annoy or change current defaults.
