# Single CLI build and required-check verification

## Result

The duplicate CLI compilation was reproduced against pre-fix Make recipes at
`c8b8e2edcfe486ed008364c1ab435aa7aa9bc846`: the regression observed two compiler/linker
invocations writing `dist/sqlite3` and failed with `AssertionError: 2 != 1`.
After the fix, a fresh CLI build invokes the compiler once, and a second identical
invocation invokes it zero times and preserves the binary bytes/mtime.

One file target now uses the same `-O2 -g`, initializer, thread, host SIMD and
linker settings as the old final direct recipe. The unused archive-based CLI
chain and its private `.libs` target were removed. `.build` remains because the
WebAssembly recipes also use it. Current static-library builds keep their own
object path. The test formerly checking the private CLI
object now checks the actual CLI executable's feature-source dependency.
The CLI tracks all included native feature sources, generated/public vendor
headers, vendor C sources and the connection initializer; the output directory is
order-only. Compiler/flag/profile changes require a fresh output directory, as
with the other native file targets. No flag-fingerprint cache was introduced.

## Checks actually run

Tests ran in an isolated checkout of the pre-fix revision with candidate Make
and test files overlaid. Native C sources came from that committed revision;
shared dirty HNSW, `.claude/`, the experiment index and BlitzRank files were
excluded. Platform: macOS 27.2 arm64, Apple Clang 21.0.0, GNU Make 3.81.

- All four packaging tests passed: CLI compilation/reuse, included-feature
  invalidation, static C consumer and standalone Rust-staged C consumer.
- The expanded CLI regression separately passed after adding checks that an
  output-directory timestamp change does not invalidate the CLI, while changes
  to the generated header, vendor sources/headers or initializer do invalidate it.
- The retired-helper reference audit caught the shared `BUILD_DIR` dependency
  in WebAssembly recipes. Its definition/directory rule were preserved, the
  complete WebAssembly section was checked against the pre-fix source, and
  `make -n wasm` passed with project-local output paths and no execution.
- All ten bootstrap boundary tests passed, including missing prerequisites,
  wrong smoke rows, output-path bounds, existing-output preservation, partial
  failure/retry, vendor failures and process-group timeout handling.
- A real fresh bootstrap passed `test-unit`, `vector-knn` and `fts5` in 14.643
  seconds, including local compiler-recording overhead. Its compiler wrapper
  recorded exactly three output-producing invocations: loadable extension,
  CLI and C-unit binary, one each. The log contained `All unit tests passed.`
- The documented existing-output Make command plus independent SQL checks took
  1.108 seconds. The wrapper recorded only the C-unit compiler invocation, the
  output contained `All unit tests passed.`, and the CLI bytes/mtime stayed
  unchanged. The independent SQLite fixture returned the expected 3-4-5 vector
  distances and FTS rows. These are single-host observations, not a comparison
  with the earlier Linux/Nix timings.

The initial packaging suite took 23.753 seconds; the final suite after preserving
the shared WebAssembly helper passed in 25.590 seconds. The expanded CLI test also
passed separately. The failing pre-fix regression took 21.309 seconds.
Timing was not an acceptance gate. Tests were not weakened to obtain a pass.

## Reproduction

Use an isolated checkout of the revision containing this fix. Supply the four
SQLite vendor inputs through the explicit vendor setup in the
[onboarding guide](../../../docs/agent-onboarding.md). Do not overwrite another
contributor's vendor inputs. Generate the header if absent, then run:

```sh
make sqlite-vec.h
PYTHONDONTWRITEBYTECODE=1 python3 tests/packaging.py -v
PYTHONDONTWRITEBYTECODE=1 python3 tests/test-bootstrap.py -v
python3 scripts/bootstrap.py dev --output /tmp/sqlite-vec-cli-check
CFLAGS=-DSQLITE_ENABLE_FTS5 make prefix=/tmp/sqlite-vec-cli-check loadable cli test-unit
```

Choose an absent output path. Confirm the full bootstrap's JSON includes all
three checks and the incremental command still prints `All unit tests passed.`
The packaging regression records real compiler arguments, asserts one initial
CLI compile/link and no repeat build, checks binary identity, and tests dependency
invalidation. Compiler discovery probes such as Linux `-dumpmachine` are excluded
from compilation counts.

To reproduce the original failure, copy the new packaging test into a separate
checkout of the pre-fix revision above, populate the vendor inputs and header,
and run `python3 tests/packaging.py Packaging.test_cli_builds_once_and_skips_unchanged_inputs -v`.
The expected failure is the two CLI output writes, not a missing prerequisite.

An independent query against the fresh built CLI used:

```sql
SELECT sqlite_version(),vec_version();
CREATE VIRTUAL TABLE v USING vec0(embedding float[2]);
INSERT INTO v(rowid,embedding) VALUES(10,'[0,0]'),(20,'[3,4]'),(30,'[6,8]');
SELECT rowid,distance FROM v WHERE embedding MATCH '[0,0]' AND k=3 ORDER BY distance;
CREATE VIRTUAL TABLE f USING fts5(body);
INSERT INTO f(rowid,body) VALUES(10,'copper otter'),(20,'silver fox'),(30,'copper fox');
SELECT group_concat(rowid,',') FROM (SELECT rowid FROM f WHERE f MATCH 'copper' ORDER BY rowid);
```

Expected complete output at the tested source is `3.45.3|v0.1.10-alpha.4`,
`10|0.0`, `20|5.0`, `30|10.0`, `10,30`. Run with `-batch -bail :memory:` and
check the exit code and every output line.

Verified vendor SHA256 values:

| File | SHA256 |
| --- | --- |
| `sqlite3.c` | `9ca336fbcbff9f1d78b4f45b6a19583fcc097192310dd2f5f6cd43b9a33d7d69` |
| `sqlite3.h` | `882ad3c0448d0324fb3a6b1a85333a9173d539ac669c9972ae1f03722ff86282` |
| `sqlite3ext.h` | `b184dd1586d935133d37ad76fa353faf0a1021ff2fdedeedcc3498fff74bbb94` |
| `shell.c` | `cfbc5b9bc4c94d8caeb3d55dc9e224038683f14a1faec5872407ba81f3b4f66a` |

## Limits and cleanup

No Linux/Windows/Nix runtime, full Rust/Lance suite or complete bundle build was
run for this repair. The Rust-staging packaging check exercised staged native C,
not Rust Cargo compilation. The caller's native Make invocation remains unchanged.
Main and `.github`/CI were not changed or dispatched. The existing dirty archive
index was preserved rather than included in this change.

Only these text instructions/results are retained in the completed trial
directory. The task-owned isolated checkout, compiler recorder, raw logs/receipts,
vendor copies and generated build outputs were removed after verification;
shared vendor inputs, caches and unrelated work remain intact.
