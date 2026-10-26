# KNN Benchmarks for sqlite-vec

## Flat-vector write-batching benchmark

After building the default extension, run from the repository root:

```sh
uv run --managed-python --project tests python benchmarks-ann/flat-batch.py --baseline /path/to/previous/vec0.dylib
```

This inserts 10,000 deterministic Gaussian float[128] vectors in one transaction
and reports insert/commit time, shadow-row changes, retained WAL bytes and mean
query time. Ten exact KNN queries are checked against an independent Python
oracle, including original-vector distances. Results must match across builds,
which run in separate processes and temporary databases. `--rows`, `--dimensions`
and `--candidate` select another workload or current library. Consecutive
inserts favor batching; interleaved reads, partitions and random updates can
flush more often. Fewer blob operations need not reduce row-change or WAL counts.

## IVF write-batching benchmark

Build the current and previous extensions with
`SQLITE_VEC_EXPERIMENTAL_IVF_ENABLE=1`, then run from the repository root:

```sh
uv run --managed-python --project tests python benchmarks-ann/ivf-batch.py --baseline /path/to/previous/vec0.dylib
uv run --managed-python --project tests python benchmarks-ann/ivf-batch.py --baseline /path/to/previous/vec0.dylib --trained
uv run --managed-python --project tests python benchmarks-ann/ivf-batch.py --baseline /path/to/previous/vec0.dylib --trained --quantizer int8
```

Each run inserts 10,000 deterministic Gaussian float[32] vectors after a
512-row seed dataset. It reports insert and commit time, shadow-row changes,
retained WAL bytes, query time and recall@10. Ten queries use an independent
Python oracle; results and original-vector distances must match across builds.
Both builds run in separate processes and temporary databases. `--candidate`
selects a different current library and `--rows` changes the insert count.
All 16 centroids are probed in trained runs. This synthetic workload does not
establish production throughput; fewer row changes need not reduce WAL bytes.

## HNSW write-batching benchmark

After `make loadable`, run from the repository root with a previous extension:

```sh
uv run --managed-python --project tests python benchmarks-ann/hnsw-batch.py --baseline /path/to/previous/vec0.dylib
```

The benchmark inserts 3,000 deterministic Gaussian vectors of 32 dimensions
in one transaction. It reports insert and commit time, total shadow-table
changes and retained WAL bytes. Each build runs in a separate process. The
check compares committed graph hashes and query results across builds, and
checks pending results against results after reopening. `--cache-mb 0`
measures immediate writes; `--rows` and `--dimensions` change the workload.
This synthetic case does not establish production speedups. SQLite already
coalesces page writes, so fewer row changes need not reduce WAL bytes equally.

## Numeric zone-map benchmark

From the repository root after `make loadable`:

```sh
uv run --managed-python --project tests python benchmarks-ann/zonemap.py
```

This creates a temporary database with 65,536 32-dimensional vectors in 64
chunks. Each chunk has one distinct numeric score. It checks filtered results
and compares warm query latency with statistics present and removed. JSON output
includes insert time, p50/p95 query latency, entry count and stats payload bytes.
Use `--baseline /path/to/previous/vec0.dylib` to compare inserts with a previous
build in a separate process. Payload bytes exclude keys and SQLite page/index
overhead. This ordered dataset measures a favorable pruning case; randomly mixed
ranges can prune less.

## Lance SQL benchmark

After building the hybrid extension, run these commands from the repository root:

```sh
uv run --managed-python --project tests python benchmarks-ann/lance.py --distribution gaussian
uv run --managed-python --project tests python benchmarks-ann/lance.py --distribution clustered
```

Each run uses 20,000 128-dimensional vectors and 100 queries with seed 42. It
compares exact `lance0` queries against the same SQL after `optimize`, checks
recall@10 against NumPy, verifies original-vector distances, and emits JSON with
latency, maintenance time, retained dataset bytes and peak Python-process RSS.
The process includes the loaded Rust extension; the separate inspection process
is excluded from RSS. The script requires Cargo: it runs `lance/examples/index-plan.rs`
to check that the committed dataset has an IVF_HNSW_SQ index and an ANN query plan.
Set `PROTOC` if the protobuf compiler is not on PATH.

The script fails if recall@10 is below 0.95. Synthetic results do not establish
recall or a speedup for production data. Dataset bytes include old versions
retained for recovery; timings use a warm cache and one process.

Benchmarking infrastructure for vec0 KNN configurations. Includes brute-force
baselines (float, int8, bit), rescore, IVF, and DiskANN index types.

## Datasets

Each dataset is a subdirectory containing a `Makefile` and `build_base_db.py`
that produce a `base.db`. The benchmark runner auto-discovers any subdirectory
with a `base.db` file.

```
cohere1m/           # Cohere 768d cosine, 1M vectors
  Makefile          # downloads parquets from Zilliz, builds base.db
  build_base_db.py
  base.db           # (generated)

cohere10m/          # Cohere 768d cosine, 10M vectors (10 train shards)
  Makefile          # make -j12 download to fetch all shards in parallel
  build_base_db.py
  base.db           # (generated)
```

Every `base.db` has the same schema:

| Table | Columns | Description |
|-------|---------|-------------|
| `train` | `id INTEGER PRIMARY KEY, vector BLOB` | Indexed vectors (f32 blobs) |
| `query_vectors` | `id INTEGER PRIMARY KEY, vector BLOB` | Query vectors for KNN evaluation |
| `neighbors` | `query_vector_id INTEGER, rank INTEGER, neighbors_id TEXT` | Ground-truth nearest neighbors |

To add a new dataset, create a directory with a `Makefile` that builds `base.db`
with the tables above. It will be available via `--dataset <dirname>` automatically.

### Building datasets

```bash
# Cohere 1M
cd cohere1m && make download && make && cd ..

# Cohere 10M (parallel download recommended — 10 train shards + test + neighbors)
cd cohere10m && make -j12 download && make && cd ..
```

## Prerequisites

- Built `dist/vec0` extension (run `make loadable` from repo root)
- Python 3.10+
- `uv`

## Quick start

```bash
# 1. Build a dataset
cd cohere1m && make && cd ..

# 2. Quick smoke test (5k vectors)
make bench-smoke

# 3. Full benchmark at 10k
make bench-10k
```

## Usage

```bash
uv run python bench.py --subset-size 10000 -k 10 -n 50 --dataset cohere1m \
  "brute-float:type=baseline,variant=float" \
  "rescore-bit-os8:type=rescore,quantizer=bit,oversample=8"
```

### Config format

`name:type=<index_type>,key=val,key=val`

| Index type | Keys |
|-----------|------|
| `baseline` | `variant` (float/int8/bit), `oversample` |
| `rescore` | `quantizer` (bit/int8), `oversample` |
| `ivf` | `nlist`, `nprobe` |
| `diskann` | `R`, `L`, `quantizer` (binary/int8), `buffer_threshold` |

### Make targets

| Target | Description |
|--------|-------------|
| `make seed` | Download and build default dataset |
| `make bench-smoke` | Quick 5k test (3 configs) |
| `make bench-10k` | All configs at 10k vectors |
| `make bench-50k` | All configs at 50k vectors |
| `make bench-100k` | All configs at 100k vectors |
| `make bench-all` | 10k + 50k + 100k |
| `make bench-ivf` | Baselines + IVF across 10k/50k/100k |
| `make bench-diskann` | Baselines + DiskANN across 10k/50k/100k |

## Results DB

Each run writes to `runs/<dataset>/<subset_size>/results.db` (SQLite, WAL mode).
Progress is written continuously — query from another terminal to monitor:

```bash
sqlite3 runs/cohere1m/10000/results.db "SELECT run_id, config_name, status FROM runs"
```

See `results_schema.sql` for the full schema (tables: `runs`, `run_results`,
`insert_batches`, `queries`).

## Adding an index type

Add an entry to `INDEX_REGISTRY` in `bench.py` and append configs to
`ALL_CONFIGS` in the `Makefile`. See existing entries for the pattern.
