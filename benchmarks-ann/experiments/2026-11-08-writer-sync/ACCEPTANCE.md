# Combined synchronization acceptance

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

Implementation: `cfa86e5` (source/FTS/vec0 synchronization). The subsequent
`ad6b7ef` changes only the wording of an earlier test result. Acceptance ran on
macOS arm64 with the existing verified default bundle; its native-library SHA256
is recorded in [consumer-result.json](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-writer-sync/consumer-result.json).

## Independent checks run in the coordinating chat

| Check | Result |
|---|---|
| Writer release suite, native extension enabled | 23 unit tests and one no-run doctest passed; two ignored subprocess helpers are explicitly executed by their parent recovery tests |
| Writer formatting | Passed |
| Writer strict all-target Clippy | Passed |
| External committed-source consumer and strict consumer Clippy | Passed |
| Read-only CLI/MCP/G0 suite from isolated committed sources and verified bundle | 76 passed in 2.43 seconds |
| Complete unmodified Rust adapter unit suite | Failed: 9 passed, 1 failed in the already-recorded timed cancellation test |
| Baseline source/harness/runner hashes, 15-case coverage and two receipt negative controls | Passed |

The external consumer separately compiles the library and calls its public API.
Direct SQL verifies actual FTS matches, vec0 KNN IDs and Euclidean 3-4-5 distances,
and canonical little-endian embedding bytes. It holds a WAL reader snapshot across
a committed source update, then verifies the new text/vector results after releasing
that snapshot. It exercises stale-source rejection, claim/undo source stability,
delete/restore/replayed-delete receipts, same-text vector-only updates and undo,
foreign-key/integrity checks and reopen. Four deliberate corruptions—wrong model,
missing lexical row, orphan state and missing state—must reject; left joins check
both missing and extra index identities/epochs. See [consumer.rs](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-writer-sync/consumer.rs).

```sh
python3 benchmarks-ann/experiments/2026-11-08-writer-sync/verify_consumer.py \
  --revision cfa86e5 --output /tmp/new-sync-consumer-result.json
VEC_WRITER_EXTENSION="$PWD/dist/bundles/949073f-default/lib/vec0.dylib" \
  cargo test --offline --locked --release \
  --manifest-path examples/write-prototype/Cargo.toml
cargo fmt --check --manifest-path examples/write-prototype/Cargo.toml
cargo clippy --offline --locked \
  --manifest-path examples/write-prototype/Cargo.toml --all-targets -- -D warnings
```

The consumer runner freezes committed writer sources, checks the trusted bundle's
hashes, and uses a fresh database/build directory. It retains its lockfile and
consumer hash, then removes the disposable work directory. Its output path must
not exist.

The read-only regression copied committed `examples/search-lab` into a temporary
root, linked the verified bundle extension to `dist/vec0.dylib`, and linked its
binary to `examples/search-lab/target/release/sqlite-vec-search-lab`. The existing
workspace venv ran `python -m pytest -q -p no:cacheprovider` over the copied lab.
[adapter-regression.json](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/benchmarks-ann/experiments/2026-11-08-writer-sync/adapter-regression.json) retains the actual output and
revision. No dirty native sources or changed tests were used.

## Unresolved regression check

This command ran without skips or altered assertions:

```sh
cargo test --offline --locked --manifest-path examples/search-lab/Cargo.toml
```

`tests::bounded_witness_cancellation_releases_transaction_and_progress_hook`
failed at `src/tests.rs:329`. Its expected error was `"cancelled"`; the actual
result had `ok:true`, `termination:"complete"`, path `[[1,null],[2,0]]`, and
`elapsed_ms:19.231792000000002`. The assertion reported `left: Null`,
`right: "cancelled"`. Final output was `9 passed; 1 failed; 0 ignored`.

The same scheduling-window failure predates synchronization, as recorded in the
[Rust review](../2026-11-07-write-history/rust-review.md#r8--p2-adapter-cancellation-unit-test-assumes-a-scheduling-window).
This is a failed full adapter unit check, not a green suite. It does not by itself
establish a runtime cancellation defect. The test was not weakened, skipped or
repeated until green; its repair remains outside this synchronization slice.

## Scope and limits

This accepts the opt-in fresh-fixture writer synchronization slice. Task title is
the searchable source; canonical supplied vectors bind to a fixed host-selected
model/preprocessing identity. Claims preserve source epochs. Undo restores canonical
values and advances revisions, rebuilding derived rows in the same transaction.
Version-1 fixtures are rejected, not migrated. Public read-only transports and C/
HNSW/Lance implementations are unchanged.

The [implementation receipt](../2026-11-07-write-history/rust-sync.md) records
stage failure/cancellation, savepoint, row-set bound, direct SQL/effect, inverse
conflict and process-exit cases. Process exit is not power-loss qualification.
There is no embedding model execution, generic application ingestion, write
transport, Lance synchronization, indexed performance comparison, other-platform
qualification, push or CI change. Existing Lance review findings remain unresolved.
