# Rust task writer prototype

An isolated library for fresh SQLite fixtures: create/claim/delete tasks, retry
committed requests, inspect history and undo changes with newer revisions. It has
no CLI/MCP routing and does not change the read-only search adapter or local bundle.

```sh
cargo test --offline --locked --manifest-path examples/write-prototype/Cargo.toml
cargo test --offline --locked --release --manifest-path examples/write-prototype/Cargo.toml
cargo clippy --offline --locked --manifest-path examples/write-prototype/Cargo.toml --all-targets -- -D warnings
```

Build prerequisites include Rust, a C compiler, libclang for Session bindings and
the lockfile's cached dependencies. Use `cargo fetch --locked --manifest-path
examples/write-prototype/Cargo.toml` once if the cache is incomplete.

## Boundary and use

```rust,no_run
use sqlite_vec_write_prototype::{Command, Mutation, Writer};
use std::{path::Path, sync::atomic::AtomicBool};

let cancel = AtomicBool::new(false);
let mut writer = Writer::create(Path::new("new-task-fixture.db"), "project", "agent-a")?;
let created = writer.apply("create-1", &Command::Batch(vec![Mutation::Create {
    id: 1, title: "Try the writer".into(), payload: None,
}]), &cancel)?;
let claimed = writer.apply("claim-1", &Command::Batch(vec![Mutation::Claim {
    id: 1, expected: created.revision,
}]), &cancel)?;
let replay = writer.apply("claim-1", &Command::Batch(vec![Mutation::Claim {
    id: 1, expected: created.revision,
}]), &cancel)?;
assert_eq!(claimed, replay);
let undone = writer.apply("undo-1", &Command::Undo { request: "claim-1".into() }, &cancel)?;
assert!(undone.revision > claimed.revision);
# Ok::<(), sqlite_vec_write_prototype::Error>(())
```

`create` reserves a new file exclusively. `open` admits only a matching prototype
application marker, project identity and schema version; it never creates a missing
file. Failed creation may leave an unusable fixture file. Existing databases are
not enrolled or migrated. The host binds project/actor identities when opening;
these are not client-supplied authorization claims.

Each `apply` acquires `BEGIN IMMEDIATE` with a 250 ms busy-retry deadline, checking
cancellation between retries and request stages. Scheduling, SQLite capture and
commit I/O have no hard latency bound. Cancellation is polled at operation
boundaries, not inside Session's native work. It rolls back before commit; cancellation
observed after a successful commit does not invalidate its receipt.

Data, monotonically increasing commit/row revisions, tombstones, typed Session
capture, changed-row index and retry receipts share one transaction. Reusing a
request ID with different arguments or actor fails. Deleting a row reserves its
identity. Undo requires every target row/tombstone still to have the recorded
after-revision, applies inverse canonical values with strict conflict aborts, then
records a new revision/history batch. It never inverts revision metadata.

Task/history reads use one read transaction. History returns at most 100 entries.
Rollback/commit failures with uncertain outcomes poison the handle: reopen and
retry the original request key to resolve the durable ledger. Do not assume an
unknown outcome means that nothing committed.

## Bounds and supported scope

- 64 distinct task IDs per batch; project/actor/request IDs: 1–128 UTF-8 bytes.
- Canonical request: 65,536 bytes; receipt/capture: 262,144 bytes each.
- Per-row title/owner/payload: 8,192 bytes combined; old and new row sets are each
  limited to 131,072 bytes. Size checks precede loading old payloads.
- Streaming capture/inversion output is bounded. SQLite's Session, pager and
  allocator work are outside that output-buffer limit; it is not an RSS guarantee.

Canonical tasks have one integer primary key, text title, nullable owner and
nullable BLOB payload. Claims require an unowned live task at the expected revision.
There is no generic SQL, schema migration, retrieval synchronization, model runner,
branching, merge or network service. All managed writers must follow this protocol;
arbitrary external SQL can bypass it. Markers/schema checks are admission checks,
not authentication or tamper-proof audit.

## Dependency isolation

The search adapter remains on rusqlite 0.41. Its Session dependency currently
requires unresolvable `bindgen ^0.73`, confirmed with offline and registry-enabled
resolution. This prototype pins the installed rusqlite 0.40.2, whose Session support
uses bindgen 0.72.1. No adapter downgrade, feature addition or dependency shim is
introduced. The writer implementation owns this temporary version split; unify it
with the adapter's dependency graph only after the Session requirement resolves and
both default read-only and writer checks pass. Check the current registry before
assuming this workstation's resolution failure still applies.

The tests cover two-connection races, exact replay after reopen, actor/key reuse,
monotonic undo, delete/restore and undo-of-undo, Unicode/BLOB/NULL values, atomic
multi-row conflicts, statement/capture/ledger failures, savepoints, cancellation,
busy waits, admission bounds, sequence overflow, schema drift, poisoned handles and
process exit before/after commit. The ignored `crash_worker` is a subprocess helper
explicitly executed twice by `process_exit_recovery`, not a skipped recovery check.
