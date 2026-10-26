# Optional metadata compression assessment

**Decision: defer general compression.** The [assessment report](REPORT.md)
records synthetic byte-size and codec-cost probes. It establishes neither
SQL speedups nor production file-size savings. No engine codec was implemented.

`recorded/` preserves the original benchmark, report generator, report and raw
results verbatim. Results include 30 stored-blob profiles, 60 zlib roundtrips,
timing batches and separately calculated length proxies and schema projections.
`manifest.json` records SHA256s. The runnable `benchmark.py` changes only input
root/output paths; it uses the current checkout and writes to a required output
path. Original source revision and runtime versions remain in the saved results.
The report's source line anchors describe that revision and can move later.

From the repository root on macOS arm64, with `uv`, `cc`, the SQLite vendor
headers and the generated `sqlite-vec.h` available:

```sh
make loadable
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python benchmarks-ann/experiments/2026-10-26-metadata-compression/benchmark.py --output /tmp/metadata-compression-replay.json
python3 benchmarks-ann/experiments/2026-10-26-metadata-compression/verify.py
```

The benchmark checks SQLite values and byte-exact codec roundtrips. It regenerates
codec measurements; the separately calculated proxies/projections stay in the
recorded results. Timing and compressed sizes can vary with runtime and platform.
The verifier checks archive hashes, syntax, profile counts and size arithmetic;
it does not claim to reproduce historical timings. Run Python with assertions enabled.
