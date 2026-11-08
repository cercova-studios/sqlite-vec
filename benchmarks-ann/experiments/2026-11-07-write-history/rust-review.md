# All-Rust review after the writer prototype

Reviewed revision `8f0af1b` with the local `rust-engineering` skill, including its
idiom, concurrency/FFI, verification and review references. The
[scope inventory](rust-review-scope.json) records all 20 tracked first-party Rust
sources and seven Cargo manifests/templates at that revision. Generated/vendor
code and concurrent follow-up benchmarks are outside this source inventory.
Build scripts, callers, unsafe boundaries, transaction state, cancellation and
resource/admission behavior were inspected directly. Code-graph coverage tools
were unavailable; this is source/manual review plus the checks below, not formal
proof or exhaustive runtime/path coverage. No unrelated production code is repaired
by this review.

## Findings, highest risk first

### R1 — P1: provider DROP deletes data before SQLite commit

Resolved by the [2026-11-08 DROP repair](../2026-11-08-lance-drop/README.md).
The following describes the reviewed pre-fix behavior.

`lance/src/provider.rs:284` routes destruction to `LanceStore::destroy`;
`lance/src/store.rs:698-703` drops SQLite metadata and immediately calls
`remove_dir_all` on the local dataset. That filesystem operation cannot roll back.

Fresh-fixture reproduction: insert/commit one `storage=lance` vec0 row, execute
`BEGIN; DROP TABLE t; ROLLBACK`, then read the restored table. The dataset directory
is already gone and the read fails with Lance's dataset-not-found error. This
blocks treating transactional DROP as safe on important local Lance data.

Smallest safe direction: retain external files when dropping SQL metadata until a
separately verified cleanup protocol can prove no committed/rollback reader needs
them. Merely deferring deletion inside an object that xDestroy frees is insufficient.

### R2 — P1: lance0 accepts duplicate rowids

`lance/src/lance0.rs:191-195` calls `LanceStore::insert` without checking logical
existence; `store.rs:265-268` stages a fresh append. Inserting rowid 1, committing,
then inserting rowid 1 again succeeds and yields both `(1,'first')` and `(1,'second')`.
Within a pending map, a duplicate can instead overwrite the earlier staged row.
Stable row identity and subsequent joins/updates cannot rely on this table's rowid.

Check existence against pending plus committed state under the SQLite writer lock
and implement SQLite conflict policy explicitly. Audit rowid-changing UPDATE for
the same missing target-identity check. The reproduction verifies duplicate INSERT;
the UPDATE extension of this finding is a source follow-up, not a claimed test.

### R3 — P1, source-contract evidence: NULL TEXT buffer can violate Rust slice safety

`lance/src/provider.rs:145-148` checks TEXT type, calls `sqlite3_value_text`, and
constructs a Rust slice without checking the returned pointer. SQLite permits that
conversion to return NULL on allocation failure, even when the initial type was
TEXT. Rust requires a non-NULL pointer even for a zero-length slice. Thus the
allowed conversion-failure path violates `from_raw_parts`'s safety precondition;
`catch_unwind` does not make an invalid slice valid.

Return an allocation/conversion error before creating a slice, and propagate it
through xPut/xSet. Inspect the error immediately as SQLite specifies. This finding
is established by the current code and primary contracts; live OOM fault injection
was not performed. Sources: [SQLite values](https://www.sqlite.org/c3ref/value_blob.html)
and [Rust slice safety](https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html).

### R4 — P2: attached schemas share unqualified Lance metadata names

`provider.rs:206-209` receives a schema name but passes only a bare shadow-table
name to `LanceStore`; `store.rs:163-171` creates/reads that name without a database
qualifier. `lance0.rs:134-136` also discards `_database_name`.

Creating `main.t` and `other.t` with different storage URIs produces only
`main.t_storage`. The second dataset is not created, and inserting into `other.t`
fails trying to open its missing dataset. Pass schema identity throughout storage
metadata lookup/create/rename/drop and quote schema/table components separately;
test same table names across attached databases.

### R5 — P2: delegated label equality ignores SQLite collation

`lance0.rs:145-156` delegates every usable label equality and sets `omit=true`.
It does not check the SQLite collation. A row labelled `Cat` disappears for
`label COLLATE NOCASE='cat'`, while the ordinary SQLite reference returns it.

Delegate only semantically equivalent collations. Other queries need correct
SQLite evaluation/fallback or an explicit unsupported filtered-KNN shape; simply
post-filtering a too-small KNN result does not restore prefilter semantics.

### R6 — P2: provider silently replaces invalid UTF-8 TEXT bytes

`provider.rs:148` uses `String::from_utf8_lossy`. Inserting `CAST(X'FF' AS TEXT)`
into an auxiliary TEXT column round-trips as `EFBFBD`, while SQLite's reference
value is `FF`. Arrow UTF-8 cannot represent that input, but silently rewriting it
is not data preservation. Reject unsupported encoding visibly or adopt an explicit
representation; never report a successful unchanged round trip.

### R7 — P2: legacy Rust C harness misses modular source invalidation

`tests/build.rs:10-11` watches only build.rs and the top-level C file, although that
file includes feature modules and the build also consumes vendor SQLite/headers.
After a clean isolated run, touching `sqlite-vec-graph.c` leaves the compiled
`libsqlite-vec-internal.a` timestamp unchanged. Cargo runs the two Rust tests
against that stale archive. Track the complete C/header dependency set, as the
main static/Rust binding build already does. The micro build script also lacks
explicit watches for its C inputs outside the package; only the legacy harness
invalidation was executed as a negative control here.

### R8 — P2: adapter cancellation unit test assumes a scheduling window

`examples/search-lab/src/tests.rs:313-329` observes the native SELECT trace, sleeps
5 ms, then interrupts and requires cancellation. On the observed failed run the
witness completed (`elapsed_ms=16.963708`) before the worker interrupted it. The
suite was 9/10, not a pass. The test needs deterministic coordination at the
execution boundary being asserted, without relabelling a VM-only cancellation
check as proof of interruption inside native loops. Do not fix it by accepting
success, skipping it, or increasing a sleep. No runtime cancellation defect is
claimed from this scheduling failure alone.

## Executable counterexamples and provenance

```sh
examples/search-lab/.venv/bin/python \
  benchmarks-ann/experiments/2026-11-07-write-history/review_lance.py \
  --extension dist/vec0_lance.dylib --output /tmp/new-rust-review-output
```

This archived runner creates only disposable fixtures. Its JSON reports violated
contracts, not passing regression tests. [Recorded output](lance-review.json)
contains five runtime counterexamples (R1/R2/R4/R5/R6) and the existing hybrid
artifact's SHA256. A fresh Lance check fails in dependencies because `protoc` is
missing; the artifact's complete build provenance was not reconstructed. The
current source implements the same implicated paths, but these reproductions are
not presented as a fresh build of revision `8f0af1b`.

## Check matrix

| Area | Actual result |
|---|---|
| New writer, debug/release | Each: 14 unit scenarios pass, one no-run doctest passes; ignored helper explicitly runs twice from recovery test |
| New writer fmt/strict Clippy | Pass |
| Read-only Python CLI/MCP/G0 suite | 76 pass in an isolated tree using the verified default native/Rust bundle |
| Default Rust adapter unit suite | Fail: 9 pass, scheduling-dependent cancellation test fails |
| Adapter fmt | Pass |
| Adapter strict Clippy, all targets | Fail: existing `collapsible_if` diagnostics at lib.rs:386 and tests.rs:303 |
| Lance check | Blocked: `lance-encoding`/`lance-file` cannot find protoc; no installation attempted |
| Lance fmt check | Fail: existing formatting differences; no bulk reformat performed |
| Rust FFI binding | `--locked` fails because generated package version differs from committed Cargo.lock; allowing lock regeneration only in the temporary tree gives two passing tests |
| Legacy Rust C harness | Two tests pass, but touching an included C module proves stale archive reuse (R7) |
| Micro all-target check | Blocked offline: `atty` required by locked criterion 0.3.6 is absent from the local index/cache |
| Simple Rust example | Source reviewed; the earlier registration task's run is not claimed as a run in this review |

The stale binding lock is a reproducibility gap: regenerate its package entry
without incidental dependency upgrades before requiring `--locked` there.
Build/lint failures remain failures, rather than a claimed clean repository gate.
The million-row/1536-dimensional Criterion workload was not executed. Neither
Miri/sanitizers nor non-macOS targets were exercised. No new MSRV support is claimed.

## Next actions

The writer prototype remains limited to ordinary fresh task tables and does not
use the affected Lance paths. Its source/FTS/vec0 follow-up can proceed separately,
with explicit source/model revision binding and independent effect checks.
Prioritize R1/R2/R3 before trusting broader Lance writes; repair the scoped SQL
semantics/build/test gaps afterward. This review does not authorize a risky cleanup
or migration of existing datasets. No push, CI change or production repair occurred.
