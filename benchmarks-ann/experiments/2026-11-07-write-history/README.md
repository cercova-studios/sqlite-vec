# P4/V1 capture and undo investigation

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

2026-11-07. No production write API or default build feature changes.
[Write/history contract](write-contract.md) defines the next prototype; it does
not enable agent writes. The existing read-only adapter remains unchanged.

## What ran

```sh
python3 benchmarks-ann/experiments/2026-11-07-write-history/run.py \
  --output /tmp/vec-write-history-replay
```

The output directory must be new. The standard-library Python runner compiles
`history.c` with the repository's vendor SQLite 3.45.3, enabling Session/preupdate
and FTS5 only in a temporary executable. No project databases are opened. Results
include source/vendor hashes, compiler/options, raw samples and check outcomes.
The independent Python SQLite reader compares every row with a deterministic model.

Three rotated-order rounds compare no capture, Session, and a transactional SQL
trigger log on identical fresh WAL databases with `synchronous=FULL`. Each case
updates 5,000 existing rows in 50 requests of 100 rows, either once or twice per row.
All modes persist the same request/actor ledger. Session persists one typed changeset
per request before commit; the SQL log UPSERTs one old/new typed row per request/key.
Neither fixture is a complete writer implementation. The log handles UPDATE only;
the Session path omits a changed-row lookup index and a production revision ledger.

## Results and limits

The frozen results are [results.json](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-07-write-history/results.json). On this macOS arm64 host:

| Updates per row/request | No capture median | Session median | SQL log median | Session DB bytes | SQL log DB bytes |
|---|---:|---:|---:|---:|---:|
| 1 | 4.275 ms | 7.640 ms | 8.549 ms | 380,928 | 311,296 |
| 2 | 4.981 ms | 8.414 ms | 13.189 ms | 368,640 | 303,104 |

These are whole-run elapsed times including capture and commits, not per-query
latency. Session is about 11%/36% faster than this log, but consumes about 22% more
database space here. Three short runs, one payload shape and one host cannot choose
a universal performance winner; filesystem/commit scheduling contributes noise.

Session's retained per-object peak is 8,656 bytes for a 100-row request. Whole-SQLite
allocator high-water readings are about 0.54–0.56 MB for both capture paths, including
pager/statement state; these are not RSS, hard memory limits or incremental costs.
Storage bytes are SQLite page counts after a truncating WAL checkpoint, without
VACUUM; BLOB overflow/page packing differs from row-log packing.

## Behavior established

The native checks validate typed old/new BLOB values, per-key net coalescing,
savepoint rollback, full rollback and all-or-nothing strict changeset conflict
handling. Explicit attachments demonstrate that unkeyed tables, NULL primary-key
rows and the virtual FTS table itself are not captured. This is why canonical table
admission and an explicit allowlist are mandatory; never attach all shadow tables.

A successful raw inverse restores the original revision number as well as the
application values. Reapplying and then changing one row causes strict inverse
application to abort, with neither row partially undone. This establishes a useful
counterexample: inversion alone is not a safe monotonic-revision agent undo API.

Four process-exit recovery cases cover Session/log before commit and immediately
after commit. The child calls `_exit` without SQLite close; a fresh independent
connection verifies integrity, task data, request ledger, history and cleared
request context. Pre-commit state disappears; committed state survives. This is
process-exit recovery, not power-loss or disk-fault injection.

## Decision

Prototype per-request Session capture plus an explicit SQL request/revision ledger.
It provides typed net changes without inventing a generic change codec or shipping
an UPDATE-only trigger benchmark as a general capture engine. Keep the SQL log as
the portability/queryability alternative. Measure full ledger/history indexing,
selective undo, inserts/deletes, request limits and two-writer concurrency before
choosing a shipped default or exposing write tools. The default bundles stay unchanged.

The first Rust prototype must use new revisions for undo, persist tombstones for
deleted identities, and store history/receipts in the same transaction as data.
Session blobs do not supply attribution, authorization, idempotency, branch/merge
semantics or model-aware index reconstruction.

Primary sources inspected: [Session overview](https://www.sqlite.org/sessionintro.html),
[table attachment and PK exclusions](https://www.sqlite.org/session/sqlite3session_attach.html),
[net changeset generation](https://www.sqlite.org/session/sqlite3session_changeset.html),
[changeset apply/conflicts](https://www.sqlite.org/session/sqlite3changeset_apply.html),
[Session memory](https://www.sqlite.org/session/sqlite3session_memory_used.html) and
[SQLite transactions](https://www.sqlite.org/lang_transaction.html). Mechanism claims
are checked against the installed vendor API and executable fixtures. No claim of
new branch/merge functionality follows from these sources or tests.
