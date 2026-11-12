# Recovery and closure verification

The immutable executable recovery commit is
`d502752999b6dad2812e89b85c801380676b499b`, on isolated parent
`812a5a38d710cdf492a430e85fcaf3da62ef1b4c`. It contains exactly 600
experiment-owned files. It excludes Nix onboarding changes, dirty HNSW work,
model weights, native libraries, SQLite databases and virtual environments.
The [README](README.md) gives exact extraction and replay commands.

## Exercised recovery

A fresh `git archive` extraction of that exact object was tested in a disposable
Python 3.12.13 environment. All 26 measured dependencies were installed from
cached wheels with `uv pip install --offline`. The recovered
`offline_replay.py` removed `OPENAI_API_KEY` and denied Python connection and
DNS calls before tests and verification. No original checkout input path,
native extension, model weights, API service or paid request was required.

The warning-strict `-B` replay passed all ten contract tests and verified:

- All 150 evaluation queries and 450 successful query/method requests.
- All 1,622 complete model permutations, upstream graph schedules and rankings.
- Independent baseline/reranked metric arithmetic and original candidate IDs.
- Exact reuse of 90 pilot receipts, all 110 pilot evidence identities and the
  original 322-entry ledger prefix; no repeated pilot inference or reset budget.
- Every continuing ledger entry maps to exactly one received response; all are
  settled, with conservative total USD 1.677428 under the original USD 5 cap.

The recovered summary SHA256 is
`a11e4b92841081a73157491521e2f113f37644c17f572eb6bc26d6d7c355cc7e`,
identical to the verified active result. Cached dependency installation precedes
the offline replay; on another host, obtaining the pinned packages may require
registry access. The tokenizer definition and small required helper sources
are contained in the recovery commit itself.

## Retention and publication boundary

The current experiment directory retains Markdown only. The active harness,
raw receipts, ignored input copies, native baseline, environment and local
calibration were relocated intact to the ignored workspace
`tmp/blitzrank-2026-11-11-retained/`. No unique evidence or shared cache was
deleted. This retained scratch is not needed for the verified Git-object replay;
its large ignored assets are not recoverable from Git and remain protected
pending any separately authorized disposal decision.

The disposable Git-object validation extraction and its environment were removed
after their result was recorded. Shared HEAD/index and the unrelated Nix/HNSW
work remain preserved. The final archival child keeps only the report,
protocol, attribution, reproduction guidance and this verification record.
Independent integration review owns any development-branch push or wiki update;
this experiment chat performed no push or default/CI change.
