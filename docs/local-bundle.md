# Use a project-local SQLite bundle

Keep this fork's SQLite CLI, search CLI/MCP adapter, native extension and C inputs
together in a project directory. The tools work after moving the bundle out of
the source checkout. Store application databases beside the bundle, not inside it.

## Build into a new project directory

From the source checkout, run:

```sh
python3 scripts/local_bundle.py build --output /tmp/my-project/.sqlite-vec
python3 /tmp/my-project/.sqlite-vec/tools/local_bundle.py verify /tmp/my-project/.sqlite-vec
```

Build prerequisites are Python 3.12+, Git, Make, a C compiler, Rust/Cargo and the
repository's `vendor/{sqlite3.c,sqlite3.h,sqlite3ext.h,shell.c}` files. Cargo builds
offline with the adapter's committed lockfile. Populate its dependency cache once
with `cargo fetch --locked --manifest-path examples/search-lab/Cargo.toml` if needed.
Running the installed CLI/MCP binaries does not require Python, Cargo or the checkout.

The builder compiles committed HEAD in a fresh temporary tree; uncommitted source
edits are excluded. The installed builder and this guide are copied from the current
checkout and included in artifact hashes. The manifest records the exact input
hashes of the native sources, vendor SQLite and adapter sources/lockfile, source
revision, platform, compiler versions, effective native flags, SQLite versions,
enabled profile and installed artifact sizes/SHA256 hashes.

`--profile default` enables FTS5 and HNSW. `--profile ivf` also enables experimental
IVF; choose a different output directory for each profile. Every build uses fresh
native and Cargo target directories, so profiles cannot reuse stale artifacts.
Build flags are controlled by the builder. Native SIMD flags target the build host;
use the bundle on a compatible OS/architecture/CPU. Cross-compilation is not supported.

Existing output paths are rejected. A failed build retains `build.log` in the new
directory for diagnosis; a complete `manifest.json` marks success. After fixing
the cause, build into another new directory. Verification checks hashes and sizes;
it is not a signature or authentication of an untrusted bundle.

## Write, query, back up and restore

Use the bundled SQLite CLI for application writes. It registers this extension
automatically and enables FTS5. These commands create new files in `/tmp/my-project`:

```sh
/tmp/my-project/.sqlite-vec/bin/sqlite3 /tmp/my-project/app.db <<'SQL'
.bail on
CREATE TABLE tasks(id INTEGER PRIMARY KEY, title TEXT NOT NULL);
CREATE VIRTUAL TABLE vectors USING vec0(embedding float[2]);
BEGIN;
INSERT INTO tasks VALUES(1,'Try the local datastore');
INSERT INTO vectors(rowid,embedding) VALUES(1,'[1,0]');
COMMIT;
SELECT sqlite_version(), vec_version();
SELECT rowid,distance FROM vectors WHERE embedding MATCH '[1,0]' AND k=1;
.backup '/tmp/my-project/app-backup.db'
SQL
/tmp/my-project/.sqlite-vec/bin/sqlite3 /tmp/my-project/restored.db ".restore '/tmp/my-project/app-backup.db'"
/tmp/my-project/.sqlite-vec/bin/sqlite3 /tmp/my-project/restored.db 'PRAGMA integrity_check; SELECT * FROM tasks;'
```

The examples are initial setup, not migrations or an idempotent ingestion API.
Choose fresh database/backup names; `.backup` and `.restore` can replace existing
destination data. For existing project data, review the intended destination first.

## Configure the read-only search adapter

The installed adapter preserves its existing CLI and stdio MCP contracts:

```sh
/tmp/my-project/.sqlite-vec/bin/sqlite-vec-search-lab --config /tmp/my-project/config.json cli
/tmp/my-project/.sqlite-vec/bin/sqlite-vec-search-lab --config /tmp/my-project/config.json mcp
```

Set the trusted config's `extension` to `.sqlite-vec/lib/vec0.dylib` on macOS or
`.sqlite-vec/lib/vec0.so` on Linux. Paths resolve against the config directory.
Keep the config and databases outside the bundle so the bundle can be replaced
independently. Use the corpus schema and requests in `docs/search-adapter.md`
inside this bundle. The adapter remains read-only; the bundled SQLite CLI is the
explicit write interface. No generic agent write tools or versioned history are added.

The native CLI uses the supplied vendor SQLite; the adapter uses rusqlite's bundled
SQLite. Both versions are recorded separately. Never share raw connection handles
between those runtimes. Other hosts must support SQLite 3.41+ for graph witnesses;
Apple's system SQLite also disables dynamic loading, so use the bundled CLI/adapter
or static per-connection Rust registration on that host.

## C consumers and rollback

`lib/libsqlite_vec0.a`, `src/sqlite-vec.c`, `src/sqlite3.c` and `include/` support
native consumers. Use `SQLITE_CORE` when compiling the C extension statically and
call `sqlite3_vec_init(db, 0, 0)` on each new connection. Compile with the profile's
native flags from the manifest when reproducing its enabled features. The static
archive contains sqlite-vec, not SQLite itself; link one matching SQLite runtime.

To update, build into a new directory, verify it and run your project checks before
switching executable/config paths. Keep the previous bundle for rollback. This
changes tools, not database contents. Data-format compatibility must be checked
before allowing a different version to write existing databases.
