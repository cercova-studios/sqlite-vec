# Lance attached-schema metadata repair

## Defect and change

At baseline `c8b8e2edcfe486ed008364c1ab435aa7aa9bc846`, both Lance entrypoints
discard the SQLite database name when opening `LanceStore`. Unqualified metadata
SQL can resolve a same-named main/temp table instead of the attached table's
version record. An attached virtual table can exist without its own metadata or
dataset, and URI/version lookup, commit, rename and DROP can target another schema.

The repair retains the callback's database name in each store and quotes database
and shadow-table identifiers separately for every metadata operation. The provider
uses that same identity when recovering a recorded dataset URI; `lance0` forwards
its previously ignored database-name argument. Reattachment under another alias
opens metadata through the new connection's alias. Provider rename qualifies the
source shadow table and keeps SQLite's required unqualified new table name.

The storage ABI, metadata keys/values, dataset format, pending/undo state and
transaction callbacks are unchanged. Existing main tables remain compatible.
No metadata migration, dataset cleanup, collation change or ranking change is
included. Metadata misplaced by older builds needs a separate recovery review;
the old records do not reliably disambiguate ownership.

## Verification — 11 November 2026

The work used an isolated local worktree based on the published baseline, excluding
concurrent Make, HNSW, index and `.claude` edits. Both hybrid builds used locked
offline Cargo dependencies, Rust/Cargo 1.99.1 and the release profile on macOS arm64.
Four SQLite 3.45.3 vendor hashes matched the earlier onboarding receipt before copy.
Official protoc 36.2 was downloaded only into task-local temporary storage and
verified against the release asset's SHA256
`83d0dc80672f53486f54299bca6177159f9cad1fd54d4d3143eeea53c4423e1b`.
No global installation or dependency/lockfile update was performed.

| Check | Observed result |
| --- | --- |
| Clean baseline hybrid build | Passed; baseline library SHA256 `55a6486bee80047d338faf3ae220bea426c7d08de188cb579baf1caacd91f09c` |
| New transaction regressions against baseline | Five setup assertion errors: attached metadata count was 0 instead of 1, for flat/HNSW/lance0 and default flat/HNSW locations; pytest exit 1 |
| Fixed release hybrid build | Passed; tested library SHA256 `e6a3a846929b76622f8a24e40126c7d7b8e311bb7f695981cb5124f0616ecfa1` |
| Complete hybrid suite | **114 passed**, including all 26 new attached-schema cases; no skips |
| Existing main-table compatibility | Candidate reopened all five baseline-created main databases with exact row/BLOB parity and successful integrity checks; broken attached databases were not migrated |
| Strict Clippy, release/all targets | Baseline and candidate both fail with the same seven diagnostics; no lint policy changed |
| Formatting | Baseline and candidate both fail on broad existing differences, starting in unchanged `lance/build.rs` |
| Changed-range whitespace | Passed |

Both release builds emitted the existing Mach-O warning that `__eh_frame` exceeds
the compact-unwind encoding limit and exception-handling performance may be affected.
The seven strict Clippy diagnostics are three `chunks_exact_to_as_chunks`, two
`needless_borrows_for_generic_args`, one `collapsible_if` and one
`doc_lazy_continuation`; they remain separate cleanup work.

The Python environment was 3.12.13, pytest 8.3.3 and NumPy 2.1.3. The new tests in
[`tests/test-hybrid.py`](../../../tests/test-hybrid.py) cover:

- Same-named main/attached tables for flat, HNSW and lance0, with distinct datasets;
  the attached alias contains spaces, a double quote and a dot.
- Explicit and default provider locations; reconnect under a different alias;
  full row/BLOB comparison against ordinary SQLite tables and independent L2
  calculations from stored coordinates.
- Read-your-writes, commit isolation, full rollback, savepoint rollback and first
  enlistment inside nested savepoints followed by inner release/outer rollback.
- Deferred-constraint failed commits, maintenance commit/rollback, provider rename
  with rollback and reopen, and full/savepoint/committed DROP with dataset retention.
- Same-named temporary tables and separate metadata in `temp`.

The first main-compatibility probe refused its fixture count because pytest's
`current` symlink duplicated a directory. Repeating the probe on the five physical
fixture directories passed every row/BLOB and integrity assertion.

## Reproduce

Use a fresh checkout containing the repair and the project's Python test environment.
Obtain missing vendor inputs explicitly with `bash scripts/vendor.sh`. Supply an
installed or task-local trusted `protoc`; this recipe does not install it globally.
The first clean Lance build compiles its dependencies; subsequent builds can reuse
the same target directory. Keep that cache separate from another agent's builds.

```sh
make loadable
PROTOC=/absolute/path/to/protoc cargo build --offline --locked --release \
  --manifest-path lance/Cargo.toml --target-dir /tmp/lance-schema-target --jobs 2
cp /tmp/lance-schema-target/release/libsqlite_vec_lance.dylib dist/vec0_lance.dylib
python -B -m pytest tests/test-hybrid.py -q
```

On Linux, use the `.so` library names. The C-only `make loadable` artifact is needed
by pytest's collection hook; the hybrid tests load the separate hybrid library.
Use the documented dependency setup if the locked Cargo cache or Python test
packages are missing; `--offline` does not provision them.

To repeat the negative control, build the baseline in a separate checkout, copy
the new test file into that checkout and select
`-k test_lance_attached_schema_transactions`. Keep the baseline library installed
there: all five configurations must fail on the absent attached metadata assertion.
Do not treat skipped tests or a missing hybrid library as a passing regression.

For the recorded lint comparisons:

```sh
cargo fmt --manifest-path lance/Cargo.toml -- --check
PROTOC=/absolute/path/to/protoc cargo clippy --offline --locked --release \
  --all-targets --manifest-path lance/Cargo.toml \
  --target-dir /tmp/lance-schema-target --jobs 2 -- -D warnings
```

These lint commands remain failures as recorded above. Native allocator/sanitizer,
power-loss/crash, other-platform and full C-only/Rust-adapter suites were not rerun
for this scoped Rust metadata fix. No main, CI, remote branch or ranking default
was changed; the commit is local only.
