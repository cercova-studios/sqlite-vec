# Comparison follow-up: SQLite filter and graph gaps

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

Measured locally on 2026-11-03. This extends the [first engine comparison](../2026-11-03-engine-comparison/REPORT.md), which covers upstream/fork sqlite-vec, Zvec and LanceDB. No production code, defaults, storage format or adapter contract changed. Keep HNSW alpha=1.0.

## Decisions supported by the results

1. **Optimize bounded graph witnesses within the existing graph module.** The bounded path repeats graph construction, scans and parent walks. Investigate budget-aware adjacency and edge-ID validation shared with existing graph code. Preserve the distinct validation policies, stable edge IDs, cancellation, work/byte accounting and transaction visibility. Do not substitute the unbounded cached path and claim equivalent guarantees.
2. **Investigate sparse exact vector access through SQLite's existing storage APIs.** A 1%-eligible flat query reads 100 times the eligible vector payload. Reuse the existing rowid mapping, incremental BLOB API and exact-ranking/filter helpers where feasible. Measure slice-read overhead against contiguous reads before choosing a dispatch policy. Preserve metadata, partition and distance predicates and pending-write visibility.
3. **Treat filtered HNSW as a SQL/extension boundary problem before tuning its graph.** Most measured latency is outside its native search callback. Keep the single `KNN_ROWID_IN` path. First assess whether existing metadata predicates can reach HNSW through a cohesive shared eligibility implementation; do not add a competing filter API, transaction-sensitive scope cache or fork SQLite's VM to avoid normal SQL costs.

These are proposed next investigations, not approved implementation specs or promised speedups. They favor the architecture's SQLite-only C core, existing pager/transaction semantics and one-way optional Rust dependencies ([architecture](../../../ARCHITECTURE.md), Builds, Optional search adapter and graph sections). No evidence here justifies a Cypher engine, SQLite fork or database migration.

## Filter profile: SIFT, 50,000 vectors, random 10% eligibility

The first comparison found fork/Zvec medians of 0.961/0.353 ms at observed recall 1.0 and ef_search=32. This follow-up profiles a fresh fork HNSW graph with the same input vectors, insertion order, M=16, ef_construction=200 and ef_search=32. Twenty warm queries precede twenty measured queries per case. The baseline and instrumented libraries query the same database and return identical ordered rows/distances.

| Case | Baseline call, ms | Instrumented call, ms | Native IN collector, ms | HNSW callback, ms |
|---|---:|---:|---:|---:|
| All | 0.0271 | 0.0384 | 0 | 0.0250 |
| Random 10% | 0.8314 | 0.8724 | 0.0425 | 0.1300 |
| Random 1% | 0.0675 | 0.0722 | 0.0045 | 0.0240 |

All values are medians; subtracting independent medians is not an exact attribution. Instrumentation overhead is 41.4%, 4.9% and 6.8% respectively. Callback timings are diagnostic, not substitutes for uninstrumented comparison results. The 10% callback records median 3,220 graph-search-distance calls and 634 membership checks. The 1% path uses the existing <=4,096-ID exact fallback; graph-search counters exclude its direct distance calls.

Separate uninstrumented SQL controls, three rounds of the same twenty queries:

| SQL control | Median ms |
|---|---:|
| Count eligible scope rows | 0.0460 |
| `? IN (SELECT id FROM scope WHERE bucket<100)` | 0.7400 |
| Same IN with RHS `ORDER BY id` | 0.9008 |
| Complete filtered KNN | 0.8546 |
| Filtered KNN with ordered RHS | 1.0221 |

The scalar IN control binds an absent ID and forces set construction without vector search. Its cost, together with the native profile, supports SQL IN materialization as the main measured cost. Adding ORDER BY makes this sample slower; it is not adopted. Plans and raw timings are retained. Neither this control nor the earlier Zvec result establishes which planner/internal operation accounts for every microsecond.

Source anchors: `vec0_rowids_in_collect` in `sqlite-vec.c` collects and sorts IDs supplied by `sqlite3_vtab_in_first/next`; `vec0_rowids_rank_exact` sets the shared small-filter policy; `hnsw_query_knn` in `sqlite-vec-hnsw.c` consumes it. Native metadata on HNSW is unsupported in this build, so the benchmark uses an ordinary SQLite side table and the existing rowid-IN boundary.

## Sparse flat-vector read amplification

One exact SIFT query has 500 eligible rows distributed across all 49 chunks. Instrumenting the flat path's full-vector-BLOB read records:

- 49 reads of 524,288 bytes: **25,690,112 bytes** delivered through `sqlite3_blob_read`.
- Eligible vector payload: **256,000 bytes** (500 × 128 × 4).
- Payload amplification: **100.352×**. All ten returned IDs pass the independent cutoff/tie oracle.

These are API bytes copied, not disk I/O or a timing attribution. SQLite page cache may satisfy the reads. The flat loop in `sqlite-vec.c` (around lines 8780–8955) builds rowid/metadata bitmaps, skips empty chunks, then reads the entire vector BLOB for each surviving chunk. Random sparse eligibility defeats chunk exclusion. This directly supports investigating sparse access; it does not prove that per-vector BLOB reads will be faster.

## Graphs: LadybugDB and SQLite

Both engines pass all 17 G0 reachability cases against the existing independent Floyd–Warshall oracle. Comparison semantics are distinct reachable nodes with minimum hop count, directed outgoing edges, zero-hop anchor inclusion and NULL-edge skipping. Ladybug results are normalized to that contract; they are not path bags. A separate Python deque BFS checks 30 anchors on a seeded random graph with 5,000 nodes/10,000 edges at depth three, over three timing rounds.

| Operation | SQLite cached legacy | SQLite bounded witness | Ladybug 0.22.1 |
|---|---:|---:|---:|
| Random graph, depth-3 reachability, median ms | 0.00598 | — | 0.0936 |
| 29-hop chain witness, median ms | 0.0134 | 9.608 | 1.737 |
| 9-hop chain witness, median ms | 0.00667 | 9.127 | 0.529 |

The chain contains 5,000 nodes/4,999 edges. Short-path timings include three calls per source/mode, including the first call; legacy later calls can reuse CSR. Legacy outputs nodes only. Bounded SQLite and Ladybug return validated edge IDs; only bounded SQLite enforces this experiment's explicit native work/byte limits. Therefore the columns do not represent equal resource guarantees or a controlled estimate of the cost of those guarantees.

In a separate 4,999-hop SQLite probe, legacy/bounded medians are 0.893/66.167 ms. Ladybug fails this deep probe at its configured 64 MiB buffer pool with `Buffer manager exception: Unable to allocate memory! The buffer pool is full and no memory could be freed!` The failure is retained, not counted as a completed timed comparison. Common shallow runs use Ladybug's depth limit 30, one worker and the same 64 MiB pool; deep runs raise the depth limit to 5,000.

Source inspection explains the bounded-path concern: `graph_witness` in `sqlite-vec-graph.c` compares each edge ID against prior IDs, scans all edges per expanded node, searches visited nodes linearly and repeats parent walks during output. It has no cache. These loops enforce existing budgets/cancellation, but duplicate work compared with the same module's CSR path. A cohesive implementation could share representation/build primitives while preserving policy differences: legacy skips NULL endpoints, whereas bounded evidence rejects invalid endpoints/IDs. This report makes no claim that a particular replacement meets all budgets or improves speed yet.

The native work-limit failure followed by a successful retry is exercised. These graph benchmarks do not rerun the full G2 cancellation/rollback suite.

## New Turso Rust and libSQL are separate comparisons

Installed isolated Python packages: `pyturso==0.8.3` (module `turso`) and `libsql==0.1.11`; reported SQLite compatibility versions are 3.50.4 and 3.45.1. Both use the previous frozen inputs: SciFact 5,183 × 384 and sampled SIFT 50,000 × 128, 100 queries each, five filter cases, three warm timing rounds. Ordinary SQL tables have B-tree bucket/band indexes. libSQL's ANN setup uses L2, max_neighbors=32, search_l=128, insert_l=200. This is one configuration, not an exhaustive engine tuning exercise.

| Engine/path, SIFT | All median ms / mean recall | Random 10% ms / recall | Random 1% ms / recall |
|---|---:|---:|---:|
| libSQL scan | 13.806 / 1.0 | 3.698 / 1.0 | 0.351 / 1.0 |
| libSQL ANN | 1.206 / 0.9953 | unsupported tested filter interface | unsupported tested filter interface |
| Turso scan, score contract fails | 14.107 / 1.0 | 4.014 / 1.0 | 0.379 / 1.0 |

libSQL's tested `vector_top_k` accepts at most three arguments; a fourth predicate argument errors. Filtered measurements therefore use eligible SQL scans, not post-filtered ANN. Fixed-predicate partial indexes and other configurations were not benchmarked. Repeated libSQL ANN calls have observed recall variation; each timed result is checked live rather than assumed identical to the quality pass.

Turso rejects the tested `libsql_vector_idx` index expression. This establishes incompatibility with that interface, not exhaustive absence of every experimental index. Turso returns **4.99704122543335** for the L2 distance from [0,0] to [3,4], instead of 5.0. Its maximum observed relative distance error is about 0.0650%; libSQL is below 0.000053%. Turso fails the common `rtol=2e-5, atol=2e-5` score check despite perfect sampled ranking recall. Its timings are diagnostic and are not a strictly equivalent scored-query winner. The numerical cause is not diagnosed and the acceptance tolerance is not relaxed.

SQLite fork HNSW, libSQL ANN and Turso scan all pass tiny same-connection own-write, savepoint rollback, full rollback and commit/reopen probes. These are lifecycle checks, not fault injection, crash recovery, isolation or durability equivalence.

## zvec-grep: host-tool assessment, performance still open

[zvec-grep's source README](https://github.com/zvec-ai/zvec-grep) and [CLI documentation](https://zvec.org/en/docs/zvec-grep/cli/) describe workspace indexing, literal/BM25/vector search, source breadcrumbs and CLI/MCP. Its [model documentation](https://zvec.org/en/docs/zvec-grep/embedding-models/) describes local embeddings and model caching. This is application-level scope, distinct from the Zvec engine already measured.

Potential product gaps are workspace extraction/chunking, freshness and literal/semantic orchestration. They belong in an optional host tool if requested. The existing Rust retrieval adapter's read-only contract should stay intact; parsers, embedding runtimes and filesystem watchers do not belong in the SQLite-only C core. No zvec-grep executable or end-to-end workspace latency/relevance benchmark was run. A fair comparison needs a shared workspace corpus and explicit chunk/model differences before interpreting results.

## Evidence and limits

[Replay instructions](README.md), [regenerated summary](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-03-comparison-followup/recorded/summary.json), raw receipts, query plans, package versions, source/input hashes and retained failures are archived together. New SQL runs comprise 2,200 quality points and 6,600 timed calls. IDs, eligibility, uniqueness, finite scores, distance tolerances and tie-aware recall are checked live against original vectors. Offline verification recomputes quality recall from frozen IDs/masks/tie sets; it cannot independently reconstruct every distance without regenerating the vector inputs. Timed receipts retain validated metrics rather than all row lists. Corrupt quality IDs/recall are explicitly rejected by the verifier.

All runs are local macOS arm64 on M5 Pro, 18 cores/48 GiB. The diagnostic native library uses the prior committed source snapshot (native code unchanged at follow-up HEAD), Apple clang 21, -O3/M1/NEON, and Python's SQLite 3.53.1. See runtime receipts for exact host versions. The unrelated dirty HNSW helper is excluded. Some diagnostic/engine workers overlapped; there is no CPU pinning or isolated cross-engine timing campaign. New runs and the earlier Zvec/Lance runs must not be combined into a precise speed ranking.

Small samples, one insertion order/build, synthetic graph structure and warm SDK calls do not establish production relevance, concurrency, million-scale, cold-cache or other-platform behavior. SQLite–Lance integration, zvec-grep runtime, hybrid application parity, crash recovery and full native/Rust/Lance/sanitizer suites remain unverified in this benchmark-only slice. Continue with a scoped design and baseline before changing native behavior.
