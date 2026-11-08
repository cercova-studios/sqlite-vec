# Local datastore roadmap and packaging checks

2026-11-07, macOS arm64, Apple clang 21.0.0. Baseline: `c269a71`.
[Current queue](../../../TODO.md) separates the five local-consumption/retrieval
priorities from the Dolt-inspired versioning track. The legacy files beside this
receipt preserve the superseded queues, including unverified historical tasks.

## Verified packaging slice

`python3 tests/packaging.py --baseline` fails all three tests on the committed
build recipes:

1. Changing `sqlite-vec-graph.c` leaves `make -q static` at exit 0 instead of 1.
2. Rust staging copies the modular source; compilation fails with
   `fatal error: 'sqlite-vec-sparse.c' file not found`.
3. `make static` omits `dist/sqlite-vec.h`.

`python3 tests/packaging.py` passes all three after the repairs. Static and
Rust-staged C consumers compile outside the checkout and run HNSW nearest-neighbor
and bounded graph-witness assertions. The dependency check also invalidates the
CLI's C object after an included feature changes. Directory mtimes are normalized
to prevent incidental directory writes from hiding missing file dependencies.
Native C inputs come from HEAD so these checks isolate build recipes and exclude
the unrelated working-tree HNSW change. This is not a C-feature regression suite.

The Rust binding was staged in the same isolated tree with its `build.rs`,
`src/lib.rs` and existing `Cargo.lock`. Public dependencies were fetched without
changing version requirements. The host-linked test compiled but failed with
`no such function: vec_version`; the reason for that host's registration behavior
was not established in that slice (resolved below). The test then
checks the registration result, version, HNSW result and bounded graph witness.
The original host-linked default and dependency requirements remain unchanged;
the separate packaging check selects `--features rusqlite/bundled` explicitly.

In that isolated tree, `cargo test --offline --features rusqlite/bundled` passed
(one test; zero doc tests).
`cargo package --offline --allow-dirty` packed 11 files and successfully compiled
the extracted package. `--allow-dirty` applies to temporary staged files; it does
not publish anything. The existing unused `p_vector` compiler warning remains.

## Replay

Prerequisites: the repository's vendor SQLite files and generated `sqlite-vec.h`,
a C compiler, Python, Make, envsubst, Rust/Cargo and cached crate dependencies
for offline Cargo checks. `make test-packaging` runs the isolated C consumers.
The baseline command is useful against `c269a71`; after committing the fix, HEAD
contains repaired recipes and is no longer a failing baseline.

For the Rust checks in this receipt, use the same isolated inputs:

```sh
python3 - <<'PY'
import importlib.util
import shutil
import subprocess

spec = importlib.util.spec_from_file_location('packaging_checks', 'tests/packaging.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
fixture = checks.Packaging()
fixture.setUp()
try:
    rust = fixture.root / 'bindings/rust'
    for name in ['build.rs', 'Cargo.lock', 'src/lib.rs']:
        target = rust / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checks.REPO / 'bindings/rust' / name, target)
    for command in [['make', 'deps'],
                    ['cargo', 'test', '--offline', '--features', 'rusqlite/bundled'],
                    ['cargo', 'package', '--offline', '--allow-dirty']]:
        subprocess.run(command, cwd=rust, check=True)
finally:
    fixture.doCleanups()
PY
```

## Limits and next work

No runtime C behavior, CLI/MCP contract, database format or existing data changed.
Other platforms, external host SQLite compatibility, the full native/adapter/Lance
suites and build-flag switching were not verified in this slice. No push,
publication or CI change. P1's project-local bundle, build manifest and clean-project
backup/restore journey remain open. History, undo, branches, merge and sync are
planned work, not capabilities established by this packaging check.

## Host-linked registration resolved

The original failure reproduced with the strengthened assertion: registration
returned 21 (`SQLITE_MISUSE`), not 0. A standalone C callback probe confirmed the
same behavior without Rust or sqlite-vec. Apple's system SQLite 3.54.0 reports
`OMIT_LOAD_EXTENSION`, rejects `sqlite3_auto_extension`, and never calls the callback.
The installed SDK's `sqlite3.h:7626` explicitly marks process-global auto extensions
unsupported on Apple platforms. Homebrew SQLite 3.54.1 returns 0 and calls it.
`otool -L` confirms that the original Rust test links `/usr/lib/libsqlite3.dylib`.

The binding now declares the C initializer's actual signature. Registration on an
open connection works with the Apple host without enabling dynamic loading. Tests
exercise version, HNSW and graph evidence after direct registration, then check
global registration separately: expected Apple rejection or successful automatic
registration on supporting hosts. The example uses direct registration and a path
dependency on this fork.

Two tests pass in each configuration in an isolated source tree:

```sh
cargo test --offline --manifest-path bindings/rust/Cargo.toml
cargo test --offline --features rusqlite/bundled --manifest-path bindings/rust/Cargo.toml
SQLITE3_LIB_DIR=/opt/zerobrew/opt/sqlite/lib \
SQLITE3_INCLUDE_DIR=/opt/zerobrew/opt/sqlite/include \
  cargo test --offline --manifest-path bindings/rust/Cargo.toml
```

These Homebrew paths describe this workstation, not portable defaults. The C probe
and SDK evidence explain the host failure; bundling SQLite is not used to hide it.
