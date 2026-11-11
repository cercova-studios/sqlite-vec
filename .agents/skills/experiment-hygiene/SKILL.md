---
name: experiment-hygiene
description: Plans and closes sqlite-vec experiments, preserving written results and reproduction instructions while removing authorized disposable artifacts. Use before research runs, at experiment completion, or for repository experiment cleanup.
---

<objective>
Keep completed sqlite-vec experiments reproducible through concise text and pinned
Git history. Remove disposable working files only after protecting active work
and checking recovery. This skill grants no deletion or publication authority.
</objective>

<quick_start>
Run from the repository root. Read [project instructions](../../../AGENTS.md),
the [architecture's experiment archive policy](../../../ARCHITECTURE.md#experiment-archive)
and the [experiment index](../../../benchmarks-ann/experiments/README.md).
Inspect `git status --short` before work. First produce an explicit keep/remove
inventory; do not start with a deletion command.
</quick_start>

<plan_a_run>
State the question, owner, fixed inputs, baseline/candidate revisions, independent
oracle, acceptance gates and resource limits before running. Keep written protocols
and results in a dated `benchmarks-ann/experiments/` directory. Use separate ignored
scratch space for datasets, databases, environments, builds and raw outputs.
Record source/input hashes, tool versions, seeds and exact commands as you go.
Obtain authorization for paid calls, downloads, external writes and data access
when required; do not discover credentials or publish private material.
</plan_a_run>

<close_a_run>
1. **Inventory and coordinate.** Identify the checkout, branch, current revision,
   dirty paths and owners of active work. Enumerate tracked and ignored experiment
   files, temporary checkouts, local bundles, build targets, environments and
   experiment-owned processes. Constrain discovery to known project locations;
   inspect any external temporary path individually. Classify every candidate as
   retained text, active dependency, disposable output or uncertain. Protect
   uncertain items until their owner or the user resolves them. Keep current
   source examples, tests, reusable tools, vendor inputs and shared development
   environments/caches unless explicitly included in the cleanup authorization.
2. **Protect active inputs and recovery.** Ask each active owner which old inputs
   it still needs. If authorized, copy them into that owner's isolated workspace;
   record source/destination, byte count and SHA256. Compare both copies, have the
   owner exercise its actual reader/import path, and obtain release of the old
   paths before deletion. Identify an immutable Git revision containing retired
   tracked harnesses, lockfiles and receipts. Verify those objects exist and
   provide an isolated-checkout recovery recipe. Git does not recover ignored or
   uncommitted files: document regeneration prerequisites, or preserve valuable
   unique data until its disposal or backup is explicitly authorized. Never
   rewrite history or purge Git objects as experiment cleanup.
3. **Retain portable text.** Keep completed directories Markdown-only: written
   protocols, reproduction instructions and result summaries. Include the
   question, tested revisions/profiles, input sources/hashes, environment, seed,
   exact commands, expected checks, observed result and limitations. Distinguish
   historical passes, negative research findings, failed checks and unverified
   claims. Pin links to retired scripts/receipts to the recovery revision; do not
   leave executable repro scripts, lockfiles, raw logs, figures or environments
   in completed directories. Update the experiment index and prepare the wiki
   summary. Preserve result tables and counterexamples; redact private material.
4. **Preview and remove only authorized paths.** Show exact targets, reasons,
   active exclusions and the recovery plan. Existing user authorization can
   cover this step; otherwise request approval for the concrete deletion list.
   Recheck ownership, Git state, file identities and protected-copy hashes just
   before removal. Stop if a candidate changed. Delete only listed paths; never
   use broad `git clean`, reset, glob-based teardown or symlink traversal. Stop
   only specific experiment processes with owner approval. Handle interruption
   by reporting what was removed and what remains. On retry, confirm absent
   targets and inspect remaining targets again; never widen the list.
5. **Verify and hand off.** Confirm retired artifact paths are absent and each
   completed directory contains Markdown only. Recheck active-copy hashes and
   owners' required inputs. Search manifests, examples, tests, docs and architecture
   for dependencies on removed paths; remove retired-only targets and replace
   live consumers' assumptions with supported setup/rebuild instructions. Check
   local/historical links, retained results and scoped tests or manifest metadata.
   Exercise a representative offline recovery in a disposable checkout when
   practical; report missing tools and failed attempts. Avoid rebuilding deleted
   artifacts in the shared tree merely to test cleanup. Review the final diff,
   stage explicit paths and preserve unrelated edits. Commit/push or update the
   separate wiki only with authorization; preserve the requested branch and CI
   boundaries. Remove your own disposable validation workspace after recording
   its results.
</close_a_run>

<success_criteria>
Report the cleanup revision, retained index/reproduction links, exact removed
scope, protected active work, checks actually run and remaining exceptions.
Distinguish logical file bytes from measured free disk space. Describe recovery
limits: tracked files remain in pinned Git history; ignored outputs require
regeneration, upstream assets/toolchains may disappear, and fresh measurements
may differ. Never present copied historical results as new experiment runs.
</success_criteria>
