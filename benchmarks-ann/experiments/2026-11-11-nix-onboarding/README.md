# Pinned native Nix onboarding comparison

## Question and scope

Does an optional pinned native environment reduce setup friction compared with
the existing host-tool route, while preserving the same contributor checks?
Two fresh agents with no prior conversation context independently followed the
project files: one used existing host tools, one used the proposed Nix route.
Neither changed C sources or weakened checks. This was one trial per route,
not a controlled estimate of agent productivity or general startup performance.

Both used C sources at `812a5a38d710cdf492a430e85fcaf3da62ef1b4c`, including the
same Makefile and bootstrap. The candidate added `flake.nix`, its generated lock
and onboarding/architecture guidance. The final tool pin is Nixpkgs
`ac588ef1c914865dad517b337ea0eb9694a4021c`, NAR hash
`sha256-XuAV2cUwaod3Bt4XFVqiXz3v1w2ex16W6Sv4S7EWres=`.
The lock's SHA256 is
`3b2ecad3a1c6314affe6e8e6feaff39b83cb12454cc640160875db494e6da553`.

## Method

Trials ran in separate Debian trixie Linux arm64 containers on the same existing
Podman VM: 8 vCPUs and approximately 4 GB RAM, hosted on macOS arm64. Containers
used image ID `37e6c80085f82a8a7f1cea2b83f036d9c084a2cd3d5be1aa6fd56d7c6f8ad7ba`,
repository digest
`docker.io/library/debian@sha256:24a01011d10eba3a023bc8585b9650b8b29021240d4217c05b905e09ef4a5232`.
Native compiles ran sequentially. Package acquisition could overlap other setup,
so acquisition times include uncontrolled network/cache effects.

Host Python monotonic clocks (`time.monotonic_ns()` for the baseline,
`time.monotonic()` for the candidate) surrounded each `podman exec`; reported
seconds include container transport overhead. Cold means empty per-container package and
Nix caches. It does not mean a cold host page cache, network mirror, VM or image
cache. VM/image preparation, authoring, agent reading/reasoning between commands
and provenance collection are excluded. Warm entry means already installed tools
and a realized Nix environment, not a new native build. Three warm entry samples
were taken; other phases have one observation. Different compiler versions can
affect build time, so this does not isolate Nix overhead from compiler behavior.

The first doctor ran before vendor acquisition on both routes. Exit 1 naming
only the four missing vendor files was expected and retained as a prerequisite
failure. Successful doctors followed explicit vendor setup. Acceptance required
the full existing C-unit/KNN/FTS bootstrap, an incremental Make check, and an
independent 3-4-5 coordinate/FTS fixture. Both agents also identified the current
queue and architecture boundaries from files, without reading private chats.

## Results

| Measured phase | Ordinary setup (seconds) | Pinned Nix (seconds) |
| --- | ---: | ---: |
| Tool/runtime acquisition | 40.417287 | 22.010295 |
| First doctor / Nix shell realization plus doctor | 0.106107 | 21.236570 |
| Explicit vendor acquisition | 0.909878 | 1.381369 |
| Warm doctor sample 1 | 0.195224 | 0.181509 |
| Warm doctor sample 2 | 0.352817 | 0.168729 |
| Warm doctor sample 3 | 0.103488 | 0.159023 |
| Warm doctor median | 0.195224 | 0.168729 |
| First full native bootstrap | 53.881174 | 79.519701 |
| Existing-output Make checks | 25.702816 | 46.886866 |

Both routes passed the complete bootstrap (`test-unit`, `vector-knn`, `fts5`),
their own independent SQL fixtures and the integrator's identical fixture below.
Source/vendor hashes matched across routes. The candidate's source and lock
hashes stayed unchanged. Interactive `nix develop` also produced a successful
doctor, and noninteractive Nix propagated an intentional command exit 7 unchanged.
Ten bootstrap boundary tests passed on the macOS host. Stable all-platform flake
evaluation passed with no builds; it warns that 26.05 is the last Intel macOS
release. No CI definition or workflow was changed or dispatched.

Nix was not faster in this trial: runtime acquisition plus first entry took
43.246865 seconds versus 40.523394 for ordinary acquisition plus doctor, and its
native builds took longer with a different compiler. The roughly 0.2-second warm
entry times include transport noise; three samples do not establish a meaningful
winner. Keep Nix optional for its pinned tools and repeatable commands, without a
speed claim. These measurements exclude agent thinking and are not end-to-end
human/agent onboarding latency.

Both agents recovered the same priorities from `TODO.md` execution order and open
follow-ups: correctness/reproducibility, indexed-write measurement, schema/recovery
and managed-writer enrollment before public writes, then retrieval work. They
identified the SQLite-only C boundary and separate read-only adapter. They did not
claim that historical writer acceptance establishes public-write readiness.

## Reproduce

Use a separate checkout of the revision containing this report and its flake.
For the historical baseline, use the exact source revision above. Record the
actual revision and any staged candidate files; do not include another agent's
dirty C sources. The [onboarding guide](../../../docs/agent-onboarding.md)
contains the normal host commands. Use fresh output directories throughout.

1. On an already available container engine, create two separate disposable
   containers from the image digest above and copy one source checkout into each
   at `/workspace`. Do not share `/nix` or apt caches between the containers.
   Keep a third authoring/evaluation container separate from the measured runs
   if regenerating the lock. Do not regenerate it during onboarding.
2. In the ordinary container, explicitly acquire tools:

   ```sh
   apt-get update
   DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 git make gcc libc6-dev gettext-base curl unzip ca-certificates
   ```

3. In the Nix container, acquire only the Nix runtime and certificate trust:

   ```sh
   apt-get update
   DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nix-bin ca-certificates
   mkdir -p /nix/store
   ```

   Debian's `nix-bin` is 2.26.3 in this trial. The disposable root container lacks
   systemd/build users; only inside that container, pass this configuration to
   each Nix invocation:

   ```sh
   export NIX_CONFIG='experimental-features = nix-command flakes
   build-users-group =
   sandbox = false'
   ```

   This test-fixture configuration is not a host installation recipe. Use the
   standard installed Nix configuration on a real contributor machine. Ensure
   candidate flake files are tracked/staged in its isolated Git checkout; Nix's
   Git source handling excludes untracked files.
4. From `/workspace`, time the ordinary doctor with
   `python3 scripts/bootstrap.py`, then `bash scripts/vendor.sh`. On Nix, time
   the first `nix develop --no-update-lock-file --command python3 scripts/bootstrap.py`,
   then `nix develop --no-update-lock-file --command bash scripts/vendor.sh`.
   Keep runtime installation separate from first environment realization.
   Run three post-vendor doctor invocations on each route and report all values.
5. Run `python3 scripts/bootstrap.py dev --output /tmp/sqlite-vec-dev` directly
   or under `nix develop --no-update-lock-file --command`. Then run the documented
   `CFLAGS=-DSQLITE_ENABLE_FTS5 make prefix=/tmp/sqlite-vec-dev loadable cli test-unit`
   in the same route's environment. This existing-output Make command still
   recompiles phony targets; label it accurately rather than claiming a no-op.

Use each built SQLite CLI with `-batch -bail :memory:` for an independent fixture:

```sql
SELECT sqlite_version(), vec_version();
CREATE VIRTUAL TABLE v USING vec0(embedding float[2]);
INSERT INTO v(rowid,embedding) VALUES (10,'[0,0]'),(20,'[3,4]'),(30,'[6,8]');
SELECT rowid,distance FROM v WHERE embedding MATCH '[0,0]' AND k=3 ORDER BY distance;
CREATE VIRTUAL TABLE f USING fts5(body);
INSERT INTO f(rowid,body) VALUES (10,'copper otter'),(20,'silver fox'),(30,'copper fox');
SELECT group_concat(rowid, ',') FROM (SELECT rowid FROM f WHERE f MATCH 'copper' ORDER BY rowid);
```

Expected rows at the recorded source revision are `3.45.3|v0.1.10-alpha.4`,
`10|0.0`, `20|5.0`, `30|10.0`, and `10,30`. Check full outputs and nonzero exits;
do not derive the expected distances from the implementation under test.

## Inputs

Both routes use the unchanged explicit vendor setup URL
`https://www.sqlite.org/2024/sqlite-amalgamation-3450300.zip`. Verify extracted
input SHA256 values before comparing results:

| Input | SHA256 |
| --- | --- |
| `shell.c` | `cfbc5b9bc4c94d8caeb3d55dc9e224038683f14a1faec5872407ba81f3b4f66a` |
| `sqlite3.c` | `9ca336fbcbff9f1d78b4f45b6a19583fcc097192310dd2f5f6cd43b9a33d7d69` |
| `sqlite3.h` | `882ad3c0448d0324fb3a6b1a85333a9173d539ac669c9972ae1f03722ff86282` |
| `sqlite3ext.h` | `b184dd1586d935133d37ad76fa353faf0a1021ff2fdedeedcc3498fff74bbb94` |

The baseline used Python 3.13.5, GCC 14.2.0, Make 4.4.1, Git 2.47.3 and envsubst
0.23.1. Compiler/version differences must remain visible when interpreting times.
The pinned shell supplied Python 3.13.17, GCC 15.3.0, Make 4.4.1, Git 2.54.0,
gettext 1.0, curl 8.23.0 and unzip 6.00. Its curl/OpenSSL used the container's
Debian `/etc/ssl/certs/ca-certificates.crt` fallback trust bundle; HTTPS trust is a
host prerequisite, not pinned by this flake. No native-source edits were included.

## Friction repaired and limits

The first unstable pin failed all-platform evaluation because Nixpkgs 26.11
removed Intel macOS support. The stable 26.05 pin evaluates all four declared
shells. This is evaluation evidence, not a native build on all four platforms.
The minimal-host guide now explicitly names Bash, curl, unzip and HTTPS trust
for vendor acquisition. Both agents encountered missing Git before tool setup;
provenance checks were deferred until the route supplied it.

The candidate's private SQL assertion initially guessed an older extension
version although SQL itself succeeded. Reading the checked-out `VERSION` corrected
that oracle error, and the exact fixture then passed. The integrator separately
checked both final CLIs against the same source-version/3-4-5/FTS oracle above.
Neither route needed human input, source patches or relaxed checks. The integrator
provided the task, container location and Nix root-container fixture configuration,
coordinated sequential builds and repaired shared guidance; these are not blind,
unassisted production onboarding trials.

The existing duplicate phony CLI recipe increases incremental build cost on both
routes. Make and C sources were kept identical for the comparison. GCC also
emitted the existing vendored SQLite `-Wstringop-overread` warning; successful
checks do not establish that warning is harmless. Python/Rust/Lance/packaging
regressions, real macOS Nix execution and model/retrieval experiments are outside
this native onboarding trial. No timing distribution or universal speed claim is
established. Preserve the ordinary route for machines without Nix.

## Teardown

After retaining these instructions and measured results, the three task-owned
authoring/baseline/candidate containers were stopped and removed. Their reported
writable layers totaled 2,505,261,524 bytes; this is not a measurement of freed
filesystem space. The newly pulled Debian trixie image and the exact temporary
trial workspace were removed, including checkouts, measurement scripts, JSON
receipts, logs, downloads and build outputs. The VM was returned to its initial
stopped state after confirming no other container was running; the task's
supervisor was stopped too. No system-wide Nix installation occurred.

Pre-existing images, containers and shared caches were preserved. Active BlitzRank
inputs, unrelated HNSW edits and `.claude/` work were excluded. Only this Markdown
record remains in the completed experiment directory. The maintained project
`flake.nix`/`flake.lock` are setup inputs, not retired experiment artifacts.
Regeneration needs the pinned source/image/tool inputs and public upstream assets;
new timings may differ. Main and `.github`/CI remain unchanged.

## Independent publication review — 11 November 2026

The Nix change was integrated onto the published BlitzRank tip `788ce61` in an
isolated checkout. The only cherry-pick conflict was the experiment index; both
result rows and the newer historical-cleanup wording were preserved. The flake,
lockfile, architecture and setup-guide bytes match the original `78f03ad` change.
No C/Make, adapter, ranking-default or CI source was changed.

The integration reviewer reran all ten bootstrap boundary tests on the macOS
host. A new container from the same pinned Debian digest independently passed
four-platform `nix flake check --all-systems --no-build --no-update-lock-file`.
On Linux arm64, the first doctor returned the expected exit 1 for exactly four
absent vendor files; explicit vendor setup and the subsequent doctor passed.
The full Nix bootstrap passed C-unit, KNN and FTS5 checks. The independent
3-4-5/FTS fixture above matched every expected row; all four vendor hashes matched,
the lock hash stayed unchanged, and the tracked source checkout remained clean.
An intentional `exit 7` propagated through `nix develop` unchanged.

These are fresh integration checks, not new onboarding timing samples or fresh
agent trials. The original comparison remains a single trial per route. Native
macOS/x86_64 builds, interactive-shell behavior and full Rust/Lance/packaging
regressions were not repeated during publication review. The Nixpkgs evaluation
warning about 26.05 being the last Intel macOS release remains visible.

After verification, the review-owned container and newly pulled pinned image were
removed. All six pre-existing images were preserved; no other container remained.
The VM was restored to its original stopped state and its review supervisor exited.
No Nix installation or configuration change was made on the macOS host.
