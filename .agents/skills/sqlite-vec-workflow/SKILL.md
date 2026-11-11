---
name: sqlite-vec-workflow
description: Set up this sqlite-vec fork as a project dependency or contributor checkout; find portable project context, fix issues, run research, and prepare wiki updates.
---

# Work with this sqlite-vec fork

Run from the repository root. Read `AGENTS.md` and
`docs/agent-onboarding.md` first; they provide the supported commands and context
map. This skill works across agent hosts and does not require private chat access.

## Choose the smallest route

1. **Consume the fork:** follow `docs/local-bundle.md`. Build committed HEAD with
   `python3 scripts/local_bundle.py build --output /absolute/new/project/.sqlite-vec`,
   then verify the installed bundle. Keep application databases outside it. Do not
   substitute upstream registry packages when the task requires fork features.
2. **Contribute:** run `python3 scripts/bootstrap.py` for read-only prerequisites,
   then `python3 scripts/bootstrap.py dev --output /tmp/sqlite-vec-dev` with a fresh
   destination. Read the reported log on failure; fix the cause and use a new
   destination. Install tools or download vendor inputs only through explicit setup.
3. **Orient:** read `ARCHITECTURE.md` and the project wiki's current decisions,
   results and roadmap. Use `HANDOFF.md`, `TODO.md` and linked receipts for the
   relevant slice. Recheck current source, branch and tests before trusting old
   status. Codex chats are optional provenance, not required context.

## Fix an issue or investigate a strategy

Inspect Git status and coordinate active file ownership before editing. Isolate
builds from another contributor's dirty sources. Reproduce a defect with a focused
regression; for research, state a question, fixed inputs and an independent oracle.
Read callers and transaction behavior before changing the implementation. Preserve
the SQLite-only C dependency boundary and existing CLI/MCP/read-only contracts.

Run relevant baseline/candidate checks. The bootstrap is a starting check, not
the full profile-specific suite. Store experiment protocols, hashes, measurements,
counterexamples and limitations in `benchmarks-ann/experiments/`; distinguish
negative outcomes from failed verification. Do not alter gates to make a result pass.

## Leave portable context

Update interface/setup docs and architecture with changed behavior. Keep historical
decisions, alternatives, experiments and roadmap in the wiki or development archive.
Use current code links and exact revisions; label implemented behavior, prototypes,
plans and unverified assumptions. Do not publish private conversations or local data.

Publication requires authorization. If granted, use the separate repository
`https://github.com/cercova-studios/sqlite-vec.wiki.git`, inspect its current pages
and concurrent changes, then commit/push only reviewed updates. Otherwise prepare
a local patch. End the handoff with changed paths, revision, checks actually run,
remaining risks and one next action. Do not infer remote-write authority from this skill.
