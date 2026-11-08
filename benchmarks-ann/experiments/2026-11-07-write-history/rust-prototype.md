# Rust claim/retry/undo prototype verification

The implementation is [examples/write-prototype](../../../examples/write-prototype/README.md).
It is an isolated ordinary-task library, not an enabled agent write interface.
The separate crate is required by a reproduced dependency incompatibility:
rusqlite 0.41/libsqlite3-sys 0.39.1 Session requires `bindgen ^0.73`, which both
offline and registry-enabled Cargo resolution reject. The prototype pins
rusqlite 0.40.2/libsqlite3-sys 0.38.2 with bindgen 0.72.1 and its own lockfile.
The read-only adapter's source, manifest and lockfile are unchanged.

## Executed writer checks

```sh
cargo fmt --check --manifest-path examples/write-prototype/Cargo.toml
cargo test --offline --locked --manifest-path examples/write-prototype/Cargo.toml
cargo test --offline --locked --release --manifest-path examples/write-prototype/Cargo.toml
cargo clippy --offline --locked --manifest-path examples/write-prototype/Cargo.toml --all-targets -- -D warnings
```

Fourteen unit scenarios pass; the README example compiles as a no-run doctest.
The ignored subprocess helper is invoked explicitly twice by the recovery test.
It exits after capture/ledger persistence before commit and immediately after
successful commit, without running Rust destructors. A new connection checks
integrity, data, revisions, requests and history, then retries the same key.
Only the committed case survives; retry produces exactly one revision/receipt.
This is process-exit recovery, not power-loss or filesystem fault injection.

Twelve synchronized two-connection claim races produce one winning claim each.
Concurrent identical requests and reopen retries return the same receipt without
another write; changed arguments/actor reject key reuse. Undo, undo-of-undo,
delete/restore and tombstones preserve newer revisions and BLOB/NULL/Unicode values.
Conservative row conflicts and strict Session conflicts roll back all rows.

Tests inject statement failures after mutation, capture and ledger stages, force
stream-output capture limits, cancel before commit at each stage, and simulate
failed commit/rollback. Failed attempts leave no request/history/version residue;
successful commit remains successful after cancellation. Unknown outcomes poison
the connection and resolve through a reopened ledger. Savepoint rollback is absent
from persisted changes. Busy waits cancel or reach their deadline. Input/row-set
limits, sequence overflow, schema drift and exclusive file creation are checked.
Foreign-key checks verify row-version/request and changed-row/request linkage.

## Read-only preservation check

An isolated committed tree uses the verified default bundle's native extension and
Rust CLI/MCP binary, excluding the unrelated dirty HNSW edit. Its Python suite passes
all 76 adapter/G0 tests, including read-only protections, parity and graph oracle cases.

The separate default Rust adapter unit suite passed 9/10. The existing
`bounded_witness_cancellation_releases_transaction_and_progress_hook` failed:
the 5 ms-delayed interruption arrived after the witness completed successfully
(`elapsed_ms=16.963708`). This is recorded as a failed check, not waived or counted
as a pass. The requested post-commit Rust review will examine this timing-based
test and other Rust paths. The prototype does not modify its implementation/test.

## Limits

Only macOS arm64 was exercised. No existing-data migration, generic table handler,
retrieval-index synchronization, Lance integration or public writer transport is
implemented. Cancellation is polled at operation boundaries and busy retries, not
inside Session capture/commit I/O. Admission/capture limits are not process-memory
limits. Complete-path performance/resource comparisons remain pending. No push/CI.
