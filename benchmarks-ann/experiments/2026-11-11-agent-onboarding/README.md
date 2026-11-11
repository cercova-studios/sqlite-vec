# Fresh-agent onboarding and bootstrap language review

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

Two independent native subagent trials use `fork_turns=none`, with no project
chat history, private memory or other agents' reports. Each starts from a fresh
local Git clone and reads the repository's agent instructions and onboarding.
Ignored vendor inputs, generated headers, build outputs and environments are
not copied from the original checkout. Host-installed tools remain available;
these are fresh project checkouts, not fresh operating systems.

## Baseline: f2098a1

The first agent downloaded the documented SQLite 3.45.3 amalgamation and cloned
the public wiki at `0405e204df0a136cfb20e587f11e9ef9bc6e1f3b`.
Native setup and version identification took approximately 80 seconds on one
prepared macOS arm64 host, measured from UTC clock observations. This is one
observation, not a general onboarding-time estimate.

The documented `dev` command passed C unit tests, in-memory exact vector KNN and
FTS5. The CLI reported SQLite `3.45.3` and vec `v0.1.10-alpha.4`. The agent correctly
identified remaining correctness/reproducibility repairs, indexed-writer
qualification, gated embedding/ANN interfaces and real proving workloads. It
distinguished public read-only retrieval, the isolated writer prototype and later
branch/merge/synchronization plans without reading a Codex conversation.

The trial exposed these concrete issues:

- A blocked download returned curl's DNS error, then unnecessary unzip/move/remove
  errors. The script continued after the first failure. Permitted network access
  recovered the download; sandbox DNS failure itself was environmental.
- `TODO.md` still described full task-only writer measurement and atomic retrieval
  synchronization as next steps. Later receipts and the wiki record their completion.
- The unit log reported `SIMD: none` despite the guide promising host SIMD flags.
  Independent Make dry runs confirmed command-line `CFLAGS=...` suppresses the
  Makefile's platform additions; supplying the same flags through the environment
  preserves the macOS arm64 NEON settings.

## Repair candidate: 0af4072

The candidate repairs vendor failure handling/staging, supplies bootstrap flags
through the environment, reconciles the queue, and points contributors to the
current execution order instead of the complete historical handoff. It also
documents the direct Make build/check path when Python is unavailable.

Two regression cases failed before their fixes and passed afterward: platform
flag preservation and stopping before extraction when curl fails. All ten
bootstrap/vendor boundary tests pass on Python 3.9.6 and 3.15.2 on macOS arm64.
They cover malformed usage, missing prerequisites, preservation of existing
output, Make path syntax, failure/retry logs, wrong smoke results, timeout cleanup,
vendor failure, repeated vendor setup and corrupt-archive cleanup. Vendor success
tests use an explicit ZIP fixture; they do not substitute for a real download.

The second fresh agent independently passed native setup and all ten boundary
tests at this exact candidate on Python 3.14.5/macOS arm64. Its unit log reports
`SQLITE_VEC_ENABLE_NEON=1`; `vec_debug()` independently reports `neon`, and the
compiler command contains `-mcpu=apple-m1 -DSQLITE_VEC_ENABLE_NEON`.

The native `dev` command took 27.033 seconds measured with `time.monotonic()`.
Clock observations cover 58 seconds from the initial clone/setup observation to
setup completion, and 94 seconds through priority inspection. These scopes differ
from the baseline's 80-second observation; they do not establish a speedup.

This agent recovered the correct priority order from the local queue and linked
receipts without consulting the wiki: correctness/reproducibility repairs,
indexed-writer qualification, then gated P2/P3 retrieval with P5 throughout. It
correctly separated supported retrieval, the writer prototype and later plans.

It also found that the native `vec_debug()` commit label identifies the last
`VERSION` change (`04d28bd...`) rather than checkout HEAD. The Make recipe confirms
this behavior. A subsequent documentation-only clarification explains the label,
requires Git HEAD/status for contributor provenance and distinguishes bundle
manifests. The evaluator read the clarification and confirmed it resolves the
trap. The queue's selective-undo wording now explicitly describes qualification
of the existing prototype, rather than reimplementation. Executable/test bytes
remain identical to the independently tested candidate.

Replay the focused boundary checks from the repository root:

```sh
python3 tests/test-bootstrap.py -v
bash -n scripts/vendor.sh
```

For an end-to-end replay, use a fresh clone of the recorded candidate and follow
the contributor guide: prerequisite report, explicit vendor setup, then `dev`
with a fresh output directory. Obtain vendor inputs through the documented
download; do not copy the developer checkout's ignored files.

## Language decision

Retain Python for the optional agent-facing wrapper, and retain Make as the
native build authority. Python 3.9+ needs only its standard library here; C-core
builds remain available directly through Make. The existing bundle builder and
test tooling also use Python, so onboarding does not introduce a new language
ecosystem into this repository.

| Choice | Fit for this interface | Cost or limitation |
| --- | --- | --- |
| Python standard library | JSON encoding, argument parsing, filesystem checks, argv-based subprocesses, timeouts and process-group cleanup are explicit and testable. | Requires an interpreter; discovery cannot run before Python exists. |
| Shell plus Make | Best direct path for the native compilation/check commands already defined by the project. The small vendor downloader remains shell. | Preserving the wrapper's JSON/error/log/timeout contract would require additional shell logic or external tools. |
| Compiled Rust, Go or C helper | Could provide a standalone agent tool if distribution becomes a demonstrated requirement. | Adds a helper build or platform-binary distribution step before contributor setup. No current need justifies that machinery. |

This is a repository-specific maintenance judgment, not a measured language speed
ranking. No alternative wrapper was implemented or benchmarked. The two tested
Python versions establish local interpreter compatibility, not availability on
every contributor machine. A future environment where Python installation is the
dominant obstacle would justify revisiting the wrapper or shipping a reviewed
binary. The direct Make path covers native work in the meantime.

The subprocess behavior is grounded in the
[Python standard-library contract](https://docs.python.org/3/library/subprocess.html).
The flag precedence finding is grounded in the project's Makefile, actual dry
runs and [GNU Make's variable precedence](https://www.gnu.org/s/make/manual/html_node/Overriding.html).

## Limits

These trials exercise native contributor onboarding and context recovery, not
the full Python/Rust/Lance suites, dependency bundle construction, missing-tool
installation, operating-system provisioning, release readiness or general agent
effectiveness. Wiki access and vendor downloads still require permitted network
access. Historical research and test receipts remain evidence for their recorded
revisions, not fresh checks of this candidate.
