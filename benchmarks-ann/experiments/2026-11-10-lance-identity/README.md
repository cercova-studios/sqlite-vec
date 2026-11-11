# Lance identity and conversion safety

`lance0` rejects duplicate logical rowids and honors SQLite's ABORT, IGNORE, FAIL,
ROLLBACK and REPLACE policies. The vec0 provider already has SQLite rowid-map
constraints, so it does not perform an additional Lance lookup on every insert.
The new lookup projects only IDs and includes pending writes and deletions.

Rowid-changing updates preserve columns SQLite marks unchanged. All rowid filter
paths use an integer cast for `i64::MIN`; Lance otherwise parses that value through
a rounded floating-point literal. Dataset format, index policy and dependencies
are unchanged. Existing datasets with duplicate IDs are not repaired.

The provider rejects NULL TEXT/BLOB conversion buffers before creating Rust slices
and preserves SQLite error codes, including `SQLITE_NOMEM`. It rejects invalid
UTF-8 rather than storing replacement characters. NULL, empty TEXT and empty BLOB
remain distinct. Failed conversions cannot stage a partial provider row.

Replacement rollback testing exposed a shared savepoint bug. When a table first
joins a transaction inside nested savepoints, SQLite announces only the innermost
savepoint. The store now records the missing outer markers at the same undo
position. Releasing the inner scope retains the history needed by an outer rollback.
This fixes both entrypoints without changing their commit protocol.

## Evidence

`receipt.json` records source, compiler, input and artifact hashes. Candidate
release builds use exported committed sources plus the explicit Lance/test edits;
the unrelated dirty HNSW file and `.claude/` are excluded. Cargo dependencies and
lockfiles are unchanged. A temporary official protoc 36.2 was hash-verified;
no global package installation was made. The previous hybrid artifact is the
hash-verified `aa33e68` build from the [DROP receipt](../2026-11-08-lance-drop/README.md),
not a baseline rebuilt during this task.

| Check | Result |
|---|---|
| Previous build: original identity/UTF-8 regressions | 12 fail for the intended behavior differences |
| Previous build: real TEXT and BLOB conversion OOM | Both terminate with SIGSEGV, return code -11 |
| Candidate before savepoint repair | 3 fail / 1 pass: UPDATE replacement rollback and first-write nested rollback on both entrypoints |
| Final hybrid and native loadable suites | 154 pass / 4 existing skips: 88 hybrid plus 66 native; three existing NumPy warnings |
| Native fault driver, ASan/UBSan and SQLite debug assertions | Both cases pass; `SQLITE_NOMEM`, no staged row, empty-value retry commits |
| Strict release Lance Clippy | Fails with seven diagnostics in existing code; no bulk lint cleanup |

`candidate-tests.txt` retains the final test output. The release build succeeds
with the existing macOS `__eh_frame` linker warning (`build.txt`).

The SQL tests compare complete identities, vectors, BLOBs and NULL values with
ordinary SQLite tables. They exercise pending/committed duplicates, multirow
conflicts, all five conflict policies, rowid replacement, integer boundaries,
savepoint/full/statement rollback, retry and reopen. The full hybrid suite retains
the DROP, reader, failed-commit and process-exit recovery regressions.

`tests/lance-fault.c` uses the real registered storage ABI in a fresh native host.
A SQL function supplies protected `sqlite3_value` arguments. A UTF-16 database
forces TEXT conversion; a nonconstant zeroblob avoids early expansion by OP_Copy.
The custom SQLite allocator fails exactly one conversion allocation, after setup.
Each case asserts the allocator was reached, the provider returned `SQLITE_NOMEM`,
no row was staged, and a later empty-value write commits. Sanitizers instrument
the C driver and bundled SQLite, not the Rust/Lance dependency graph; leak detection
is disabled. The ordinary Python hybrid tests also run both fault cases.

An exploratory combined run pointed the C-only loadable suite at the hybrid
library. Its exact module-list assertion correctly failed because `lance0` is
present. `hybrid-native-module-mismatch.json` retains that result (153 pass / 1 fail /
4 skips). Final validation uses a fresh C-only library for that suite and the
candidate hybrid library for the hybrid suite; the assertion is unchanged.
Failure logs with significant whitespace are JSON-encoded to preserve their
exact output. Six installed-artifact replacement/savepoint/OOM checks also pass.

## Replay

From the repository root, with Rust, a trusted protoc, bundled SQLite headers/C
source and the test Python environment installed:

```sh
PROTOC=/absolute/path/to/protoc cargo build --release --locked --manifest-path lance/Cargo.toml
cp lance/target/release/libsqlite_vec_lance.dylib dist/vec0_lance.dylib
make loadable
tests/.venv/bin/python -m pytest -q tests/test-hybrid.py tests/test-loadable.py --tb=short
```

The library suffix is `.so` on Linux. The Python fixture compiles and runs the
native fault driver automatically. On macOS, the sanitizer check is:

```sh
cc -O1 -g -DSQLITE_DEBUG -fsanitize=address,undefined -fno-omit-frame-pointer \
  tests/lance-fault.c vendor/sqlite3.c -I. -Ivendor -o /tmp/lance-fault-sanitized
ASAN_OPTIONS=detect_leaks=0 /tmp/lance-fault-sanitized \
  "$PWD/dist/vec0_lance.dylib" /tmp/fresh-text-data.lance text
ASAN_OPTIONS=detect_leaks=0 /tmp/lance-fault-sanitized \
  "$PWD/dist/vec0_lance.dylib" /tmp/fresh-blob-data.lance blob
```

Use a new dataset location for each replay. Linux also needs `-ldl -lpthread -lm`.
No other platform, remote dataset, legacy duplicate repair, new bundle release,
adapter/writer suite, push or CI change is included. Attached-schema qualification
and collation semantics remain open in TODO.md.
