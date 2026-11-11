# Indexed writer synchronization verification

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

The follow-up to `8f0af1b` extends the isolated Rust prototype, not the read-only
adapter, native C core or shipped bundle. Scope and current API are documented in
[the indexed fixture contract](../../../examples/write-prototype/README.md#indexed-source-fixtures).

## Native input and checks

Tests use the verified default bundle's vec0 library from source `949073f`, excluding
the unrelated working-tree HNSW edit:

`dist/bundles/949073f-default/lib/vec0.dylib`

SHA256: `4785d6f2ef2b1da23ed8c99f218b9dd519f4270e0789a3a20d52f8727f4c0261`.

```sh
python3 scripts/local_bundle.py verify dist/bundles/949073f-default
export VEC_WRITER_EXTENSION="$PWD/dist/bundles/949073f-default/lib/vec0.dylib"
cargo fmt --check --manifest-path examples/write-prototype/Cargo.toml
cargo test --offline --locked --manifest-path examples/write-prototype/Cargo.toml
cargo test --offline --locked --release --manifest-path examples/write-prototype/Cargo.toml
cargo clippy --offline --locked --manifest-path examples/write-prototype/Cargo.toml --all-targets -- -D warnings
```

Executed on the indexed candidate: 23 unit tests and one no-run doctest pass in
debug and release; strict all-target Clippy, formatting and bundle verification
pass. The two ignored helpers are each explicitly invoked twice by their recovery
tests, so all four process-exit cases execute. These checks include the original
task-only prototype scenarios. The read-only adapter's code and dependencies are
unchanged; its earlier 76-test transport/G0 pass and timed Rust-test failure remain
recorded in the [prototype receipt](rust-prototype.md) and [Rust review](rust-review.md).

The indexed tests compare direct SQL FTS rows and MATCH results, real vec0 KNN L2
distances against direct float64 norms, canonical little-endian bytes, task/source/
index epochs, immutable receipts, foreign-key checks and integrity checks. A held
WAL reader retains old text/vector rows during a writer commit. Reopened fixtures
observe the complete new state. Same-text vector replacement and undo preserve
the intended vector rather than inferring change from text alone.

Two-connection claim/retry tests use separate native-loaded connections. Claim and
undo-claim tests assert no FTS/vector checkpoints and exactly five ordinary SQLite
changes per commit: task, version, sequence, request and changed-row entry.

Injected failure and cancellation after canonical mutation, FTS, vector, capture,
ledger and before commit preserve the old source and both projections. A second-row
failure proves rollback of a completed first row. A malformed native vec0 insert
after FTS writes, capture-output exhaustion, savepoint rollback, strict multi-row
inverse conflicts and uncertain commit handling exercise independent failure paths.
Process-exit helpers run explicitly before commit during vector publication and
after successful commit, then reopen, check integrity and retry the original key.

Negative controls remove FTS/vector/state rows, inject orphan state, mismatch model
and epoch, truncate canonical vector bytes and store a 1 MiB canonical BLOB. Reads
and affected mutations reject corruption; size/dimension checks precede canonical
vector materialization. Seven 4,096-dimensional canonical rows fit row-set admission;
eight exceed it, proving vectors count toward the existing 131,072-byte total.

Testing found and fixed a real counterexample: undo-of-restore deletes canonical
rows before derived-state rebuild, and Session checks foreign keys at apply completion.
A derived `index_state` foreign key rejected this valid transient state. The derived
projection now uses explicit identity/model/epoch checks without that foreign key;
the ordinary canonical vector/task relationship retains its foreign key. Undo of
restore and repeated inverse restore are regression cases.

## Limits

This is a fresh-fixture prototype on macOS arm64. It does not migrate version 1
fixtures, expose write transports, run embedding models, synchronize Lance storage,
or repair unmanaged SQL writes. Cancellation remains stage-polled; input/capture
limits do not bound native RSS or power-loss behavior. The rusqlite Session version
split and existing all-Rust review findings remain open. Indexed performance must
be measured separately from the task-only complete-path baseline. No push or CI.
