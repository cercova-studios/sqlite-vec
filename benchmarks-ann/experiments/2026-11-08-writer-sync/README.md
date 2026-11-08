# Full writer/history measurement

Measured the committed task writer `8f0af1b7cf6fef3396ff66dc05bb5244b8341d0a`
through its public Rust API on macOS arm64, before retrieval synchronization.
The preferred receipt is [baseline-locked.json](baseline-locked.json). It records
the generated measurement lockfile, exact source/harness/runner hashes, compiler,
SQLite version, database settings, individual timings, storage sizes and child
process resource accounting. [baseline.json](baseline.json) is the preliminary
successful run; it did not retain its generated measurement lockfile and is not
the reproducibility reference.

## Replay

From the repository root, with the cached Rust dependencies:

```sh
python3 benchmarks-ann/experiments/2026-11-08-writer-sync/summarize.py
python3 benchmarks-ann/experiments/2026-11-08-writer-sync/run.py \
  --revision 8f0af1b \
  --output /tmp/writer-baseline-replay.json
```

The output must not exist. The runner copies committed writer sources into a
fresh temporary directory, builds the measurement binary offline in release
mode, and removes its temporary build/databases on exit. It never builds against
the dirty working tree. The receipt retains the generated Cargo lockfile; the
build uses `--locked` after offline resolution. Summary validation checks source,
harness and runner hashes and rejects missing observations or a failed oracle.

## Workload and results

Three repetitions alternate the order of five fresh-process cases: batch/payload
sizes `(1,0)`, `(16,0)`, `(64,0)`, `(16,512)` and `(64,128)`, in rows/bytes per row.
Each case performs 50 sequences of create, claim, identical retry, stale-revision
rejection, undo claim, delete, undo delete and history read. Each mutation timing
includes request validation, revision checks, Session capture, row-version and
changed-row indexes, request/receipt persistence and commit. SQLite uses WAL and
`synchronous=FULL`. Separate history probes add 1,000 committed requests per case.

The table pools 150 observations per operation. Percentiles use the sorted sample
at `floor((n-1)*p)`; they are descriptive measurements without confidence intervals.

| Operation | 1 row, no payload p50 / p95 ms | 64 rows, no payload p50 / p95 ms |
|---|---:|---:|
| Create | 0.095 / 0.168 | 0.724 / 0.787 |
| Claim | 0.096 / 0.172 | 0.774 / 0.844 |
| Retry | 0.009 / 0.018 | 0.021 / 0.027 |
| Reject stale revision | 0.007 / 0.010 | 0.012 / 0.014 |
| Undo claim | 0.110 / 0.191 | 0.768 / 0.817 |
| Delete | 0.096 / 0.169 | 0.732 / 0.802 |
| Undo delete | 0.108 / 0.175 | 0.755 / 0.798 |

Peak Rust child RSS ranges from 5.64 to 10.16 MiB across all cases. It includes
the driver, SQLite, measurement code and independent SQL oracle, not just the
writer; it is not a memory bound. With 64 rows and 128-byte payloads, create
p50/p95 is 0.825/1.654 ms. Median final database sizes range from 446,464 to
6,103,040 bytes; retained canonical request, capture and receipt bytes are broken
out in the receipt, alongside DB/WAL/SHM sizes while connections remain open.

### History scaling finding

For a single task, `history(id,100)` p50 grows from 0.0118 ms at 7 retained
entries to 0.0498 ms at 105 and 0.2487 ms at 1,005. The current query filters
`changes` by ID, joins `requests`, and orders by request revision. Its output cap
does not establish bounded history-processing work. This is a measured follow-up
candidate; no history index or query optimization is included in this experiment.

## Correctness and negative controls

Outside timed calls, an independent SQL reader checks canonical title/owner/BLOB
values, live state, row revisions, total sequence, request count and changed-row
count after every sequence. Public retry receipts must agree exactly; stale
requests must reject. Each case checks integrity, foreign keys and reopen state.
History probes check output size and newest revision. All 15 cases passed.

The summary validates complete case/sample coverage. Two deliberate receipt
mutations—a failed oracle and a removed timing observation—must both be rejected.
These controls validate receipt acceptance, not recovery or retrieval semantics;
the synchronization implementation needs its own failure/corruption controls.

## Failed probes and limits

The first harness used a table-valued `pragma_wal_autocheckpoint`, which SQLite
rejected (`no such table`). The final harness reads direct PRAGMAs. macOS
`/usr/bin/time -l` failed under the sandbox with `sysctl kern.clockrate:
Operation not permitted`; final RSS uses a fresh Python wrapper's
`RUSAGE_CHILDREN`, with exactly one Rust child. On macOS `ru_maxrss` is bytes.

A proposed 16-row case with 4,096-byte BLOB payloads correctly returned
`Error::Limit`: its canonical JSON byte arrays exceed the existing 65,536-byte
request limit even though raw row data fit. The recorded cases use admitted
payload sizes; product limits were not changed. Failed probes are not counted
as completed measurements.

This is a sequential local writer baseline, not an optimization comparison.
Filesystem caches were not flushed; the first operation is retained and no
warmup is discarded. It excludes embedding generation, FTS/vector synchronization,
transport, concurrent contention, power loss and other platforms. Process exit
and correctness coverage for the writer itself are recorded separately in the
[prototype receipt](../2026-11-07-write-history/rust-prototype.md).

## Synchronization follow-up

The subsequent opt-in source/FTS/vec0 implementation is committed as `cfa86e5`.
See [combined acceptance](ACCEPTANCE.md) for independent consumer and regression
results, including the unresolved pre-existing Rust adapter cancellation-test
failure. These baseline timings exclude the new indexed path and do not establish
its performance.
