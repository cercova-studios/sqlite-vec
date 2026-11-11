# Working on sqlite-vec

Start with [the portable onboarding guide](docs/agent-onboarding.md). It covers
dependency use, contributor setup and project context for any agent host.
Run `python3 scripts/bootstrap.py` for a read-only prerequisite report.

## Find the contract

- Read [ARCHITECTURE.md](ARCHITECTURE.md) before design or nontrivial edits. Use
  its named sections; this document does not currently use numbered sections.
- Read the [wiki](https://github.com/cercova-studios/sqlite-vec/wiki) for decisions,
  rationale, research outcomes and roadmap. [HANDOFF.md](HANDOFF.md),
  [TODO.md](TODO.md) and linked experiment receipts retain local development
  context. Recheck current source and revision before relying on historical results.
- For a guided workflow, read
  [.agents/skills/sqlite-vec-workflow/SKILL.md](.agents/skills/sqlite-vec-workflow/SKILL.md).
  No Codex account, private memory, global skill installation or chat access is
  required to follow the repository guidance.

## Preserve the boundaries

Keep the C core dependent only on SQLite. Optional Rust adapters, Lance storage,
model runners and experiments must not become native-core dependencies. Preserve
SQLite transactions, savepoint rollback, read-your-writes and explicit error
propagation. Treat ordinary application data as canonical; derived indexes must
match it. The existing retrieval CLI/MCP adapter stays read-only unless a separate
write interface is explicitly authorized and verified.

Inspect `git status` before editing and before committing. Other agents may work
in this checkout. Coordinate overlapping files, preserve unrelated edits and stage
explicit paths. Use an isolated checkout when another agent is changing sources
needed for your baseline. Never discard their work to make a build pass.

## Finish with evidence

Reproduce issues before fixes and test the affected public interface. The native
bootstrap runs C unit and disposable in-memory vector/FTS checks; it does not
replace profile-specific Python, Rust, packaging or hybrid regressions. Report
failures and skips as such. Never claim an old receipt as a fresh check.

Update current contracts beside code. Keep plans, detailed validation records and
historical decisions in the wiki/development archive. Make that context usable
without private conversations. Link exact sources and distinguish implemented
behavior, prototypes, negative results and planned work. Wiki publication, remote
writes, important-data migrations and destructive operations require authorization;
guidance and suggested commands do not grant it.
