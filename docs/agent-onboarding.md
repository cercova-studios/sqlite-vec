# Start working with this sqlite-vec fork

Use this page from any agent host or terminal. Project context lives in files and
the [project wiki](https://github.com/cercova-studios/sqlite-vec/wiki), so access to
Codex conversations is optional. Start with the route that matches your job.

## Choose a route

| Job | First action | Working contract |
| --- | --- | --- |
| Use this fork in another project | Build a local bundle below | [Bundle guide](local-bundle.md) |
| Fix an issue or contribute | Run the contributor bootstrap below | [Architecture](../ARCHITECTURE.md) |
| Research a search strategy | Read the wiki and the closest experiment's report | [Experiment archive](../benchmarks-ann/experiments/) |
| Continue existing work | Read the wiki roadmap, then the relevant receipt | [Handoff](../HANDOFF.md) and [queue](../TODO.md) |

The upstream registry packages listed in the README do not identify this fork's
committed work. Pin the fork revision when you need its features.

## Build a contributor environment

The automated path supports macOS and Linux. It needs Python 3.9+ (standard library
only), Git, Make and a C
compiler available as `cc`. A fresh checkout also needs `envsubst` (gettext) to
generate `sqlite-vec.h`. Rust, models and a server are unnecessary for C-core work.

1. Clone the development branch into a new directory. Check its current status
   in the wiki before choosing work; create your own branch before editing.

   ```sh
   git clone --branch codex/dev-local-datastore https://github.com/cercova-studios/sqlite-vec.git
   cd sqlite-vec
   git switch -c codex/my-task
   ```

2. Inspect prerequisites. This command is read-only and works from any current
   directory when invoked by its absolute path.

   ```sh
   python3 scripts/bootstrap.py
   ```

   A successful report has `ok: true` and `missing: []`. Missing prerequisites
   produce `ok: false`, exact names and a guide path. Install missing tools with
   your host's package manager explicitly. For a **fresh checkout** without
   `vendor/`, review and run `bash scripts/vendor.sh`: it downloads SQLite 3.45.3
   from sqlite.org and populates ignored vendor inputs. Do not rerun it over
   another contributor's vendor files. The vendor script downloads and extracts
   in a temporary directory and stops on download/HTTP/extraction errors before
   copying into `vendor/`. Retain the SQLite source identity when comparing
   results. No bootstrap command installs tools or downloads data.

3. Build and verify the working-tree C sources into a fresh directory.

   ```sh
   python3 scripts/bootstrap.py dev --output /tmp/sqlite-vec-dev
   /tmp/sqlite-vec-dev/sqlite3 :memory: 'SELECT sqlite_version(), vec_version();'
   ```

   Success reports C unit, exact vector KNN and FTS5 smoke checks. The bootstrap
   enables FTS5 and uses Make's host SIMD settings. It writes build outputs and
   `bootstrap.log` into the selected directory; Make may generate the ignored
   root header. Each Make invocation has a 600-second timeout; the smoke query
   has 30 seconds. These checks establish a starting environment, not a complete
   regression pass. Output paths accept ASCII letters, digits, `/`, `.`, `_` and
   `-` because Make interprets target names. Existing destinations are rejected.

   Record `git rev-parse HEAD` and `git status --short` for source provenance.
   Native `vec_debug()`'s `Commit` label comes from the last change to `VERSION`
   in the Make header recipe; it may precede checkout HEAD. Working-tree edits
   are included in this contributor build. A bundle instead records its compiled
   revision and input hashes in `manifest.json`.

After a failure, read the full reported log, fix the cause and choose a fresh
output directory. Completed outputs are retained. For incremental C work, reuse
the successful directory directly:

```sh
CFLAGS=-DSQLITE_ENABLE_FTS5 make prefix=/tmp/sqlite-vec-dev loadable cli test-unit
```

This direct Make command is also the native build/check route when Python is
unavailable, after vendor files and `envsubst` are installed. Make creates the
output directory. It prints build/test output directly and does not produce the
bootstrap's JSON result, retained log or timeout handling. Set `CFLAGS` in the
environment as shown so Make can append its host SIMD flags.

For Python regressions, install `uv` explicitly and follow the repository's
`tests/pyproject.toml` (Python 3.12+). The native suite expects `dist/vec0`, so build
that location before running it. Some tests require experimental or hybrid builds;
record skips and run the profile relevant to your change.

```sh
make loadable
uv run --managed-python --project tests pytest -q tests/test-*.py
```

For adapter work, follow the [fixture and checks](../examples/search-lab/README.md).
For Lance, follow its [build and transaction guide](../lance/README.md); the
hybrid target additionally requires Rust and protoc. Use the existing
`make test-packaging` when changing native packaging. Avoid `make install` for
onboarding: project-local artifacts are sufficient.

For changes to the bootstrap itself, run `python3 tests/test-bootstrap.py -v` and
exercise the `dev` command in an isolated checkout with a fresh output directory.

## Use the fork as a dependency

Reuse the existing builder. It builds **committed HEAD**, excluding working-tree
source edits, into a new destination. Read [bundle prerequisites and recovery](local-bundle.md)
first: Python 3.12+, Rust/Cargo and a populated offline Cargo cache are required
in addition to native tools and vendor files. The guide also shows how to copy
the bundled starter app into the new project and verify actual writes, vector
queries, rollback, reopening and backup/restore.

```sh
python3 scripts/local_bundle.py build --output /tmp/my-project/.sqlite-vec
python3 /tmp/my-project/.sqlite-vec/tools/local_bundle.py verify /tmp/my-project/.sqlite-vec
/tmp/my-project/.sqlite-vec/bin/sqlite3 :memory: 'SELECT vec_version();'
```

Keep application databases outside the bundle. Use the bundled SQLite CLI for
writes or integrate the native library/C inputs using the guide. The included
search CLI/MCP adapter remains read-only. Its [configuration and fixture](../examples/search-lab/README.md)
define the supported corpus contract; it is not an arbitrary SQL tool. Verify a
new, disposable application database before opening important data. Bundle
manifests identify inputs and artifacts; hashes do not authenticate untrusted code.

## Load project context without reading every chat

Read [AGENTS.md](../AGENTS.md), this page and the relevant sections of
[ARCHITECTURE.md](../ARCHITECTURE.md). Start with the
[queue's execution order](../TODO.md#execution-order) and open follow-ups for
current priorities. Read only the relevant heading/receipt in the long handoff;
its older sections are history, not the current queue. Read the wiki's
[architecture](https://github.com/cercova-studios/sqlite-vec/wiki/SQLite-First-Architecture),
[decisions](https://github.com/cercova-studios/sqlite-vec/wiki/Technical-Decisions),
[evidence](https://github.com/cercova-studios/sqlite-vec/wiki/Evidence-and-Experiments)
and [roadmap](https://github.com/cercova-studios/sqlite-vec/wiki/Roadmap) for the
project-wide briefing. Read only the report and source for the slice you
will change. Source and current checks decide behavior; historical receipts say
what was tested at their recorded revision and platform.

The project started from upstream's small C vector extension and expanded toward
a reusable local SQLite datastore. The original retrieval roadmap led to ANN,
filtered/hybrid retrieval, graph traversal and a read-only agent adapter. Later
work added relocatable bundles and an isolated versioned-write prototype. The
goal is to extend SQLite through ordinary tables, virtual tables and explicit
host APIs while preserving its transaction and single-writer model.

| Area | Current boundary and evidence |
| --- | --- |
| Native core | SQLite-only C; exact vector scans and optional native indexes. [Architecture](../ARCHITECTURE.md#builds) records storage and rollback behavior. |
| Agent retrieval | Separate Rust CLI/MCP adapter; host-selected corpus and read-only access. [Contract](../examples/search-lab/README.md) describes bounds, exact flat/L2 search and graph operations. |
| Optional Lance | Separate hybrid build/storage provider. Read the [identity/conversion receipt](../benchmarks-ann/experiments/2026-11-10-lance-identity/README.md) before changing it. |
| Versioned writes | [Fresh-fixture prototype](../examples/write-prototype/README.md), separate connection and lockfile; public write tools are not enabled. |
| Research | [Experiment archive](../benchmarks-ann/experiments/) retains specifications, oracles, reports and negative outcomes; these are not all production features. |

The wiki and queue carry changing priorities. Do not infer completion from a
checkbox alone. The [Rust review](../benchmarks-ann/experiments/2026-11-07-write-history/rust-review.md)
has follow-up repair receipts; read those before treating an old finding as open.
The [late-interaction experiment](../benchmarks-ann/experiments/2026-11-08-late-interaction/README.md)
rejected native MaxSim for its measured configuration. Preserve negative results
instead of reviving an approach because its report is hard to find.

## Contribute, investigate and hand off

1. Inspect `git status --short --branch`, active collaborators and the current
   queue. Reserve files before overlapping edits. An isolated clone/worktree
   keeps another agent's dirty sources out of your builds. Never sweep unrelated
   files into a commit.
2. Reproduce an issue or state a research question before changing behavior.
   Trace callers, transaction state and persisted formats in current code. Keep
   the C core independent of optional Rust, model and remote-service dependencies.
3. Make the smallest complete change. Run the relevant regression against the
   baseline and candidate; preserve failure output, skips and platform limits.
   Experiments need fixed inputs, an independent oracle and a report explaining
   whether the result supports adoption.
4. Update current interface/setup docs and architecture with the code. Put
   decisions, alternatives, experiment results and roadmap changes in the wiki
   or designated development archive. Do not place private chat transcripts,
   workstation paths or credentials in public documentation.
5. Hand off the exact revision, changed paths, checks actually run, remaining
   limits and one concrete next action. With publication authorization, edit the
   separate `sqlite-vec.wiki.git` repository, check links and push only reviewed
   pages. Otherwise prepare a local wiki patch and report its location. A chat
   summary is not a substitute for maintained portable context.

The bootstrap applies [AXI principles](https://axi.md/): a read-only default,
noninteractive commands, compact JSON, explicit missing prerequisites and
recoverable errors. Its stdout contains one JSON result (except human-readable
`--help`); full build diagnostics live in the reported log. Exit codes are 0 for
success, 1 for prerequisite/build/check failure and 2 for invalid usage. Existing
bundle and adapter command contracts remain as documented in their own guides.
