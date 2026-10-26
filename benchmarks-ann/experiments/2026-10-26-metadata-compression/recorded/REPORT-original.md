# Optional metadata compression assessment

2026-10-26. Assessment only; no engine, schema, dependency or CI changes.

## Recommendation

Defer general metadata compression until a real workload demonstrates that metadata storage is a material constraint. If a narrower experiment is warranted, start with opt-in, lossless integer packing for low-cardinality or narrow-range values in read-mostly tables. Leave Boolean bitmaps, general floats and TEXT unchanged in that first slice. This is a proposed scope, not an implemented feature.

The current engine already stores Booleans as one bit per row. Integer and float point reads and writes touch eight bytes. Whole-chunk compression would add decoding to those reads and replace small in-place writes with chunk re-encoding and a variable-length SQL BLOB update. Filtered KNN already reads full metadata chunks, so that path is a more natural fit.

## Measured probes

Unchanged vec0 produced real metadata blobs for ten synthetic distributions at chunk sizes 128, 1,024 and 4,096. SQLite results matched all inputs; zlib levels 1 and 6 decoded every blob byte-for-byte (30 stored profiles, 60 codec roundtrips). These are warm CPU codec microbenchmarks, not query or update benchmarks. Zlib is an experimental probe; it is not selected as a production dependency.

The table uses level 1 and assumes a hypothetical 16-byte framing overhead. Raw fallback chooses the smaller payload. A production frame and API have not been designed. Times are medians of nine batches of 20 calls after five warmups, including Python binding/allocation overhead.

| Distribution, 1,024 rows | Raw chunk B | Chosen payload B | Payload reduction | Encode us | Decode us |
|---|---:|---:|---:|---:|---:|
| integer_categories | 8,192 | 1,210 | 85.2% | 10.56 | 4.64 |
| integer_sequence | 8,192 | 1,601 | 80.5% | 22.40 | 5.79 |
| integer_random64 | 8,192 | 8,192 | 0.0% | 34.12 | 0.59 (raw fallback; no decode needed) |
| float_random | 8,192 | 7,939 | 3.1% | 46.70 | 13.89 |
| float_constant | 8,192 | 100 | 98.8% | 3.83 | 2.63 |
| boolean_balanced | 128 | 128 | 0.0% | 4.07 | 0.11 (raw fallback; no decode needed) |
| boolean_sparse | 128 | 60 | 53.1% | 2.06 | 0.25 |
| text_categories | 16,384 | 1,599 | 90.2% | 13.68 | 4.72 |
| text_random12 | 16,384 | 10,915 | 33.4% | 75.01 | 21.40 |
| text_random32 | 16,384 | 8,866 | 45.9% | 46.84 | 17.67 |

Encode timings show the cost of trying the codec even where raw fallback wins. Decode timings on fallback rows describe the probe only and would not be incurred by a correctly tagged raw block. No end-to-end speedup is established.

The long-TEXT case also stores 32,768 bytes of full strings outside the 16,384-byte chunk. Compressing its prefix views alone saves 7,518 bytes out of 49,152 bytes of metadata payload: 15.3%, not the 45.9% reduction in the prefix chunk. SQLite B-tree/index overhead is additional.

Integer frame-of-reference packing has a smaller theoretical payload on the low-cardinality sample: 537 bytes for four bits per value, including an assumed 16-byte frame, 8-byte base and 1-byte bit width. The sequential sample needs 1,305 bytes; random 64-bit integers need 8,217 bytes and should remain raw. These are size calculations, not implemented or timed integer codecs.

At 4,096 rows, an integer/float chunk is 32 KiB and a TEXT view chunk is 64 KiB. The categorical integer probe decodes in about 16 us; random float decoding takes about 53 us. A whole-chunk design would incur that work for an eight-byte point result unless it caches decoded data or uses independently encoded blocks. Cache lifetime and rollback would then become additional correctness obligations.

## What it could save

Illustrative schema: one categorical INTEGER, one random FLOAT, one balanced BOOLEAN and one categorical TEXT column. Across 1,024 rows, metadata payload falls from 32,896 to 10,876 bytes (66.9%) under the level-1 probe plus assumed framing and raw fallback. This is a synthetic example, not the user's production schema.

| Float32 vector dimensions | Reduction in vector + metadata payload |
|---|---:|
| 32 | 13.43% |
| 128 | 3.95% |
| 768 | 0.69% |

These estimates exclude rowids, validity, zone-map entries, indexes, SQLite page slack and TEXT overflow tables. They are not database-file savings. A separate 100-row SQLite BLOB-table length proxy is recorded in the JSON: it demonstrates page granularity, without pretending a codec has been integrated into vec0. No file-size or query-latency improvement has been verified in the engine.

## Compatibility and failure boundaries

- Existing numeric point reads perform direct offset reads without a codec header check. An older extension could interpret compressed bytes as numeric values. A new format must be rejected at table-open time by unsupported readers; an extra info-table flag alone is insufficient. A new opt-in constructor declaration is one possible boundary because current constructors reject unknown options. Syntax and migration policy remain undecided.
- Do not silently rewrite existing tables through optimize. Existing tables should retain their format; any future conversion needs an explicit, separately approved migration and recovery plan.
- Preserve all 64 integer bits, signed extremes, exact float data and existing TEXT comparison behavior. Quantizing metadata would change filter semantics and is outside scope.
- Bound decoded sizes by the table's chunk size and column type before allocating. Reject malformed frames, offsets, lengths and truncated payloads; corruption is not a reason to reinterpret a compressed payload as raw.
- Keep zone-map bounds conservative and transactionally consistent with metadata writes. Failed writes, ROLLBACK TO, full rollback, deleted-slot reuse, compaction, rename and reopen must retain current semantics. Any write buffer or decoded cache needs explicit lifetime and rollback rules.

## Reference design versus optional quality dials

A reference design would use independently encoded, typed blocks with raw fallback, an explicit format declaration and bounded decoding. Point reads would decode one block; filters could stream blocks; writes would replace a bounded block through SQLite transactions. Exactness, corruption handling and rollback are mandatory. Block size and additional codecs are quality dials that must be chosen using end-to-end workloads. This design is a target for evaluation, not a commitment to a new storage abstraction.

A whole-chunk generic wrapper is smaller to implement, but transfers cost to point reads, updates and dependency management. The C core currently depends on SQLite and ships as a single source; Python's installed zlib does not make zlib a free C-core dependency. A compact integer-only codec could avoid that dependency, but still needs format negotiation and update behavior.

Before implementation, collect a real schema/data sample and its read/write mix. Measure per-table page use, file size, cold and warm filtered KNN, point reads, random metadata updates, WAL traffic, CPU and peak memory under the same workload. Decide acceptable tradeoffs before selecting codec/block size. Include incompressible data, INT64_MIN/MAX, corrupted frames, allocation failures and rollback fault injection. Preserve existing index/metadata eligibility restrictions.

## Source anchors and reproducibility

- [Architecture: C-core dependency boundary and zone maps](/Users/rohit/sandbox/sqlite-vec/ARCHITECTURE.md)
- [Chunk layout sizes](/Users/rohit/sandbox/sqlite-vec/sqlite-vec.c:5536)
- [Numeric point metadata reads](/Users/rohit/sandbox/sqlite-vec/sqlite-vec.c:5250)
- [Metadata filter reads and exact-size checks](/Users/rohit/sandbox/sqlite-vec/sqlite-vec.c:8442)
- [Metadata writes and zone-map maintenance](/Users/rohit/sandbox/sqlite-vec/sqlite-vec.c:10823)
- [Constructor bounds/default and unknown-option rejection](/Users/rohit/sandbox/sqlite-vec/sqlite-vec.c:6238)
- [Benchmark script](/tmp/vec-metadata-compression-assess.py)
- [Raw measurements, timing batches and length-proxy page counts](/tmp/vec-metadata-compression-results.json)

Run from the repository: `UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python /tmp/vec-metadata-compression-assess.py`. The script regenerates codec measurements; the additional length-proxy fields and illustrative projection in the saved JSON were calculated separately in this assessment.

Environment: {"head": "c85e736b2e7ef008fbfac0733b0de96ad7ed99ef", "platform": "macOS-27.0.1-arm64-arm-64bit", "machine": "arm64", "python": "3.12.13", "sqlite": "3.53.1", "zlib": "1.2.12", "timing": "9 batches of 20 calls after 5 warmups; CPU codec microbenchmarks, not SQL timings", "sqlite_pages": "dbstat bytes for the raw metadata shadow table; excludes other tables/indexes", "scope": "synthetic full chunks from the unchanged vec0 engine; byte-exact roundtrips; no production codec or format change"}.

Limits: synthetic full chunks only; no production data, partially filled chunks, complete codec implementation, SQL latency comparison, RSS profile, cross-platform validation or prospective-format fault tests. Existing application tests were not rerun for this read-only assessment. Graph coverage for sqlite-vec.c is partial around preprocessor lines; relevant metadata source was read directly. Shared source notes were inaccessible in this sandbox. No new commit, dependency, push or CI change.
