# Fresh-agent dependency onboarding

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

A native subagent started with `fork_turns=none`: no prior project chat context,
private memory or other agents' reports. It used an isolated clone of committed
`c1fde29d23c12efcdf7d66e8f315f8ca49f5ef02`, followed the dependency guide, and built
a new project-local default bundle. It downloaded SQLite through the documented
vendor script and did not borrow vendor files, headers, bundles, build outputs or
environments from the original checkout.

The prepared macOS arm64 host already had Python 3.14.5, Cargo 1.99.1 and native
tools. Shared Cargo source dependencies were cached; every bundle used a fresh
Cargo target directory. This is a fresh-project test, not a clean-OS or cold-cache
installation benchmark.

## Baseline result

The complete default bundle built successfully, including the native CLI/library,
C inputs and read-only Rust CLI/MCP adapter. The guide was sufficient; no installer
defect blocked the dependency path. The only failed setup step was sandbox DNS,
which recovered through permitted network access to the documented SQLite URL.

The agent created a small Python standard-library consumer beside `.sqlite-vec`.
It uses the installed CLI, not Python SQLite extension loading or an upstream
registry package. Ordinary task records and a flat vec0 table share IDs. An
independent `math.dist` oracle verifies IDs `101,202,303,404` and L2 distances
`0,5,10,13`. Separate CLI processes verify persistence/reopening, read-your-writes,
full rollback, savepoint rollback, backup/restore and integrity.

The complete consumer was relocated and the original source path renamed. The
consumer and installed manifest verification still passed from outside the source
checkout. The coordinating agent independently reran the app and all five existing
bundle checks, including adapter CLI/MCP discovery and static C consumers.

The agent recommended retaining the checked consumer beside the guide. This is a
guidance improvement, not evidence that the prior one-vector SQL example failed.

## Packaged starter candidate

Candidate `60877e84b6800587e067ef362d56569a3bd95031` preserves the consumer as
[`examples/bundle-consumer/app.py`](../../../examples/bundle-consumer/app.py) and
installs it as `examples/app.py` in the bundle. The installed guide shows how to
copy it beside `.sqlite-vec` and run it. Each invocation retains fresh disposable
databases outside the bundle; existing runs are not overwritten. An explicit
`-O` refusal prevents false success with disabled assertions.

One actual example defect was reproduced during integration: absolute backup paths
quoted inside SQLite dot commands failed when the project directory contained an
apostrophe. Running the CLI from the database directory with fixed relative backup
names fixes the issue without embedding paths in commands. The full installed
journey now runs in a pathname containing both spaces and an apostrophe.

The same agent checked out the exact candidate in its isolated clone and built a
new full bundle. It reused only downloaded vendor inputs and the Cargo source
cache. Using only the installed README, it copied the packaged starter into a new
project, ran it from outside the checkout, relocated the project and reran it.

| Candidate check | Actual result |
| --- | --- |
| Full default build | Passed; 40.242 seconds around the complete builder subprocess using `time.monotonic` |
| Installed copy/run, relocation and hash verification | Passed; expected IDs/distances, reopening, rollback, savepoints and restored KNN |
| Updated `tests/bundle.py --bundle PATH -v` | Five tests passed, no skips; unittest elapsed 2.558 seconds |
| CLI/MCP/C journey | Describe/search/fetch, write rejection, MCP initialization/discovery, static archive and amalgamated C consumers passed |
| Failure/integrity controls | Tamper/size/missing artifact, traversal/symlink, existing output and failed-build completion-marker checks passed; `-O` refused |
| Coordinating-agent acceptance | Installed verifier, starter on Python 3.9.6, and equality of repository/installed/copied starter bytes and manifest hash passed |

Native SQLite is 3.45.3; the adapter bundles SQLite 3.54.1. Native flags are
`-DSQLITE_ENABLE_FTS5 -mcpu=apple-m1 -DSQLITE_VEC_ENABLE_NEON`. The manifest records
the runtimes and artifact hashes separately. Timings describe this prepared host;
they do not establish a cold-install SLA or compare language implementations.

## Replay

Follow the [bundle guide](../../../docs/local-bundle.md) from an isolated clone,
obtain the documented vendor inputs and populate Cargo dependencies explicitly if
needed. Build into a fresh project directory. Use the installed guide's copy/run
commands, then run the existing integration suite from the source checkout:

```sh
python3 tests/bundle.py --bundle /absolute/project/.sqlite-vec -v
```

Run the copied starter with normal Python, without `-O`. Move the complete project
and repeat the app and installed verification to check source independence.

## Limits

No Linux/Windows, cold Cargo cache, IVF/Lance, broad ANN quality/performance or
production-data compatibility was tested. MCP coverage is initialization and tool
discovery; it does not establish every MCP operation's behavior. Compatible
OS/architecture/CPU remain required. Hash verification detects changes against a
trusted manifest and does not authenticate an untrusted bundle. Raw local reports
and workstation-specific paths were retained outside the repository.
