# Lance DROP rollback repair

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

`LanceStore::destroy` removes SQLite version metadata and retains local/remote
dataset files. Both the vec0 provider and `lance0` use it. Files remain after a
committed DROP because rollback and existing readers can still need them. This
patch adds no deletion queue, filesystem cleanup, transaction hooks or new ABI.

## Failure and fix

The original provider calls `remove_dir_all` inside xDestroy. `BEGIN; DROP TABLE;
ROLLBACK` restores SQLite's schema but cannot restore the dataset. A WAL reader can
also outlive a committed DROP and still need the files.

Testing exposed two more failures in the same lifecycle:

1. vec0 freed its internal state even when xDestroy failed. Its statement pointer
   could also refer to an already finalized statement. Denying version-table DROP
   through SQLite's authorizer crashed the fresh baseline process with SIGSEGV.
   Cleanup now retains the vtab on failure and clears the finalized pointer before
   helper calls that can fail.
2. Successful destruction loses a provider's pending map. Even refusing destruction
   after vec0 has dropped shadow tables can trigger schema reload and lose buffered
   values. The provider now rejects DROP with uncommitted writes/maintenance before
   any other vec0 shadow table changes. Commit or roll back that transaction first.
   SQLite's xDestroy dispatch does not forward module-specific error text, so the
   host may report a generic SQL error. This restriction is explicit in the
   [current contract](../../../lance/README.md#transactions).

`lance0` now implements its destroy callback through the same store method, including
removing its SQLite version table. The C core change preserves the live object on
failed DROP for ordinary vec0 tables as well.

## Executed checks

Fresh release builds use committed `8f643ae` sources exported with `git archive`,
then only the candidate's two Rust files and `sqlite-vec.c` replace baseline inputs.
The unrelated dirty HNSW file and `.claude` are excluded. Cargo's existing build
cache is reused; baseline and candidate artifacts are separate, hash-recorded files.
Cargo dependencies and lockfiles are unchanged.

The host had no protoc. An official macOS ARM protoc release was downloaded into a
temporary directory and verified against the release asset's SHA256 before use.
[receipt.json](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-lance-drop/receipt.json) records its version, URL/digest, compilers, source/vendor
hashes and artifact hashes. No global package installation was made.

| Check | Result |
|---|---|
| Fresh baseline DROP/reader controls | 12 fail / 3 pass: provider file loss and missing `lance0` metadata cleanup |
| Fresh baseline failed-DROP lifetime control | SIGSEGV, subprocess return code -11 |
| Candidate full hybrid suite | 70 pass, no skips |
| Candidate default `test-loadable.py` | 66 pass / 4 existing skips: three TODO, one slow |
| C unit harness | Pass |
| C unit harness with ASan + UBSan | Pass, including flat/HNSW failed-DROP lifetime regressions |
| Strict release Lance Clippy | Fails: same seven diagnostic code/message pairs on baseline and candidate |
| Lance formatting check | Fails: existing crate formatting differences; no bulk formatting applied |

[Negative controls](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-lance-drop/negative-controls.json), [baseline JUnit](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-lance-drop/before-rollback_and_readers.json),
[hybrid JUnit](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-lance-drop/after-full.xml), [default JUnit](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-lance-drop/after-core-full.xml) and the
[Clippy comparison](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-lance-drop/clippy-comparison.json) retain the actual outcomes. These are
failure controls and scoped passes, not a globally green repository check.
Raw baseline JUnit and formatter output are JSON-encoded to preserve their exact
whitespace without adding whitespace violations to the patch.

The regression fixtures cover flat, HNSW and `lance0`. Direct ordinary SQLite rows
provide the independent BLOB/NULL/vector oracle. Tests compare complete dataset-file
hash maps and SQLite schema/version rows, not just cached query answers. They check
full rollback, savepoint rollback, deferred-FK commit failure, reopened connections,
subsequent writes, primed/unprimed WAL readers and committed DROP. Six subprocess
cases exit without destructors before/after commit and reopen to verify recovery.
Pending insert/update/delete/optimize cases verify rejection, own reads, savepoints,
commit/rollback and a subsequent successful DROP. Authorizer failures before and
after provider metadata removal verify atomic schema recovery and continued use.

## Replay

Use a clean checkout of the repair commit and the project's test environment:

```sh
export PROTOC=/path/to/protoc
cargo build --release --locked --offline --manifest-path lance/Cargo.toml
make hybrid loadable
tests/.venv/bin/python -m pytest -q tests/test-hybrid.py tests/test-loadable.py
make test-unit
make test-unit CFLAGS='-g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -DSQLITE_VEC_ENABLE_NEON'
```

The sanitizer flags above were exercised on macOS arm64. Use the appropriate SIMD
configuration on another platform. Build dependencies must already be cached for
`--offline`; `make hybrid` also needs protoc on PATH or in `PROTOC`. The normal test
environment includes NumPy and Syrupy; the search adapter's smaller Python
environment lacks NumPy and cannot run the full hybrid suite.

Initial test setup also needed the default vec0 library for `tests/conftest.py`.
An attempted metadata-column failure fixture was rejected because Lance storage
does not support metadata columns; the final test uses supported authorizer failure
points instead. No production admission limits or failure expectations were relaxed.

## Limits

Verified on macOS arm64 only. Process exit is not power-loss or filesystem-fault
validation. Remote object storage was not exercised. Dataset retention consumes
disk until its owner establishes that no table/database/reader needs the files;
automatic garbage collection requires a separate design. Other Lance review
findings remain open. No push or CI changes.
