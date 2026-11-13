# Use this fork from Rust

Add this dependency to your application's `Cargo.toml`. Replace
`<full-commit-sha>` with the reviewed 40-character fork revision:

```toml
[dependencies]
sqlite-vec = { git = "https://github.com/cercova-studios/sqlite-vec", rev = "<full-commit-sha>" }
rusqlite = { version = "0.41.0", features = ["bundled"] }
```

The committed Cargo package includes its C amalgamation, extension header, SQLite
3.45.3 headers and licenses. Cargo needs a C compiler; consumer builds require no
Make, Python, gettext or vendor download. A clean clone also supports
`cargo build --manifest-path bindings/rust/Cargo.toml`.

Register the extension on each open connection:

```rust
let db = rusqlite::Connection::open_in_memory()?;
let rc = unsafe {
    sqlite_vec::sqlite3_vec_init(
        db.handle().cast(), std::ptr::null_mut(), std::ptr::null(),
    )
};
assert_eq!(rc, rusqlite::ffi::SQLITE_OK);
```

The binding compiles sqlite-vec, not SQLite. With the configuration above,
rusqlite supplies the application's single bundled SQLite runtime. Headers are
compile-time inputs, not a second runtime. Pass a live connection from that same
library and check the registration result before issuing queries. Never pass a
connection handle from another SQLite runtime.

Per-connection registration also works with Apple's system SQLite, which rejects
process-global auto extensions. Callers using `sqlite3_auto_extension` retain that
option on supporting hosts. The C initializer takes three pointer arguments and
returns a SQLite result code; a bare zero-argument invocation is invalid.

## Database compatibility

`tests/rust-packaging.py` checks flat float32/L2 vec0 files exchanged with the
registry release `sqlite-vec = "=0.1.9"`. Separate executables use bundled
rusqlite 0.41.0, so the two extensions never share a process. Checks cover numeric
metadata filters, auxiliary text, exact distances, committed writes, savepoint
and transaction rollback, and reopening with both versions. Fork-specific HNSW
tables are outside this interchange contract.

Before allowing 0.1.9 or another older writer to modify a fork-written database,
remove `zonemap_*` entries from each affected vec0 `_info` table while the fork
still controls writes. Older writers do not maintain those bounds. Stale bounds
can cause later fork queries to omit matching rows. Missing bounds use normal
filtering. Qualify upgrades and downgrade procedures on a disposable database copy
before switching an application's writers; these fixtures do not establish
compatibility for every schema, vector type or historical release.

## Maintain the package

After changing native sources or the header template, regenerate the committed
inputs from the repository root. This maintainer route needs the normal native
prerequisites and the SQLite 3.45.3 vendor files:

```sh
make -B sqlite-vec.h
make -C bindings/rust deps
```

Keep the package version in `Cargo.toml` equal to root `VERSION`. Commit the
regenerated C/header files and licenses beside the manifest. Cargo never
regenerates them during a consumer build. The freshness check compares them to
the current modular sources and header template.

Cache the locked registry inputs once, then run the regression suite:

```sh
cargo fetch --locked --manifest-path bindings/rust/Cargo.toml
cargo fetch --locked --manifest-path tests/rust-compat/Cargo.toml
python3 tests/rust-packaging.py -v
```

The suite needs Python 3.11+, Git, Cargo and a C compiler. It checks package
freshness, bundled-rusqlite 0.31 registration, an unpacked Cargo archive, and a
clean revision-pinned Git consumer with bundled rusqlite 0.41. Registry builds
use cached dependencies; Git fetching uses only a disposable local repository.
The `Rust package` workflow runs the same suite on Linux x86_64 and macOS arm64
for pushes and pull requests to `main` and `codex/dev-local-datastore`. CI checks
the committed inputs without regenerating them, so stale amalgamations or headers
fail the freshness check.
The existing [example](../../examples/simple-rust/demo.rs) selects this fork by path.
