# Late-interaction reranking: plan and specification

Status: L0/L1 executed in this archive; the relevance gate failed. See
[the measured result](REPORT.md). L2–L4 remain proposals; no native scorer,
writer integration or transport capability was added.
Date: 2026-11-08. Baseline: `8f643ae` on `ann-hnsw`.

## 1. Outcome and scope

Add an optional, bounded second-stage reranker that compares contextual token
vectors using MaxSim. Keep existing BM25, dense-vector and reciprocal rank fusion
(RRF) candidate generation and all default search behavior intact. Model inference
stays in an explicit external producer. The SQLite-only C extension owns numerical
scoring only if experiments justify a native implementation.

First delivery: a reproducible text-only experiment with an independent scorer,
fresh SQLite fixtures, pinned model artifacts, real relevance labels and measured
resource costs. A native scalar function and adapter integration are later gates.
PDF/image-patch models, token-vector ANN indexes, training, compression, network
services and existing-data migrations are separate follow-ups.

The reference design is exact, revision-consistent MaxSim over the complete
admitted candidate set, with streaming candidate processing and explicit budgets.
Fewer candidates, shorter model inputs and quantized vectors are quality dials:
each changes the admitted search space or scores and requires separate evidence.
Engineering effort is not a reason to silently turn those dials down.

## 2. Current architecture and affected boundaries

Root `ARCHITECTURE.md` has descriptive headings, not numbered domain/extension/
invariant sections. The applicable sections and current code are:

| Boundary | Current contract | Consequence for this proposal |
|---|---|---|
| Builds; C core | SQLite-only C; Rust may depend on C, never the reverse | No model runtime, Rust or Lance dependency in a scorer |
| Optional search adapter; `examples/search-lab/src/lib.rs` | Shared `Adapter::call`, exact flat/L2, BM25/RRF, trusted scope, one transaction per operation; 64 MiB database admission | Candidate/rerank snapshot and corpus-size qualification need explicit integration |
| `sqlite-vec-rescore.c::rescore_knn` | Quantized candidate scan followed by single-vector float rescoring | Reuse the staged-search idea; do not reinterpret rescore as token MaxSim |
| Experimental task writer; `examples/write-prototype/src/retrieval.rs` | One canonical vector per source; separate task/source revisions; atomic projections and canonical undo | Multi-vector artifacts need a separate representation and history design |
| `examples/write-prototype/README.md` | 65,536-byte requests, 131,072-byte row sets, 262,144-byte capture output | Do not insert large matrices into the existing command/capture path or raise these limits implicitly |
| `reference.yaml`, `tests/test-loadable.py`, Makefile | SQL inventory, registration coverage and modular C/amalgamation builds | Any new SQL function must update these owners and supported build variants |

Current source identity is host-scoped: `(corpus, signed-int64 document ID)`.
The writer's `versions.source_revision` is a content/vector/liveness epoch, distinct
from claim revision. The search adapter's current document schema has no general
source-revision binding; one must be designed explicitly before integrating it.

Active work: `TODO.md` prioritizes local datastore safety and P2/P3/P4 integration.
It lists general reranking research, not a committed late-interaction feature.
During authoring, `lance/src/lance0.rs`, `lance/src/store.rs`, `tests/test-hybrid.py`,
`sqlite-vec.c`, `tests/test-loadable.py`, `sqlite-vec-hnsw.c` and `.claude/` carried
unrelated edits. L2 therefore overlaps actively edited registration/test boundaries
and needs ownership coordination before implementation. This document edits none
of those files.
The existing adapter cancellation-test failure and Session dependency split remain
recorded blockers; neither is resolved by writing a new scorer.

## 3. Pipeline and scoring contract

```text
explicit model producer -> pinned query token matrix
                                  |
eligible BM25/dense/RRF candidates + matching document token artifacts
                                  |
                   exact bounded MaxSim -> score-descending results
```

For query rows `Q[0..m)` and document rows `D[0..n)`, each with dimension `d`:

`score(Q,D) = sum_i max_j dot(Q[i], D[j])`

V1 score kind is `maxsim_normalized_dot_v1`; higher scores are better. Inputs are
model-produced float32 rows with float64 squared norm within `1e-4` of one.
Normalization occurs explicitly in the producer under the pinned profile. The
scorer rejects invalid normalization; it does not silently normalize or clamp.
Float32 rounding means this contract is normalized dot product, not an assertion
of mathematically exact cosine normalization.

The scalar reference and first native version preserve accumulation order and
avoid fast-math reassociation. A later SIMD reduction needs explicit numerical
compatibility qualification; tolerance tests do not override this v1 contract.

Rules:

1. Accumulate each dot product and the outer sum in float64, in token/component
   order. Initialize maxima below every valid similarity; all-negative matches
   must remain negative. Do not initialize maxima to zero.
2. Preserve query-token multiplicity. Repeated document vectors do not change a
   maximum. Padding and masked document rows are removed by the producer; query
   augmentation/special-token policy belongs to the exact model profile.
3. Reject zero rows, NaN, infinity, wrong dimensions, malformed lengths, empty
   query matrices and empty document matrices. A missing document representation
   is not a zero score: zero could incorrectly outrank a negative valid score.
4. Sort final results by score descending, then signed document ID ascending
   within one corpus. First-stage rank is retained as provenance, not silently
   fused into MaxSim. Return the first `min(limit, candidate_count)` rows.
5. An empty eligible candidate set returns a successful empty result after query
   admission. Missing/stale/model-incompatible matrices in a nonempty candidate
   set fail the entire late-interaction operation with affected IDs.

“Exact” means every admitted token pair was evaluated. It does not mean exhaustive
corpus retrieval, exact real-number arithmetic, or guaranteed semantic relevance.
No reranker can recover a relevant document absent from its candidate set.

## 4. Model, matrix and identity contracts

Dense first-stage and late-interaction profiles are distinct. They need not use
the same model or dimensions. Never interpret a pooled embedding as token vectors.

An immutable profile identifies model/weight revision and hashes, tokenizer and
processor revisions, query/document prefixes, tokenization and augmentation,
masking, input truncation/chunking, dimensions, normalization and score kind.
Record model/data licenses and the inference runtime/version in provenance.
Changing any representation-affecting field creates a new profile identity.

The first experiment selects one publicly available text checkpoint only after
verifying its license, offline artifact availability and official scoring behavior.
No package, checkpoint download or model execution is authorized by this spec.
Synthetic matrices establish numerical correctness before model integration.

Matrix v1 payload: contiguous row-major IEEE-754 float32 **little-endian** bytes,
with dimensions/token count/profile stored in ordinary metadata. No subtype,
JSON float array, native-endian assumption or custom binary header is required.
Validate `bytes == checked(4 * token_count * dimensions)` before allocation/read.
The native function derives row count from length and the supplied dimension;
the host additionally checks the stored count and profile dimensions.
Artifact metadata also records role (`query` or `document`) and the SHA256 of exact
UTF-8 source input before profile preprocessing. Require the correct role at each
pipeline position; query and document encoding policies can differ within a profile.

Document identity is `(corpus, document ID, current source epoch)`; representation
identity also includes the late-interaction profile. Query identity includes its
profile and exact encoded input. Record document/query truncation explicitly:
truncated model input is not silently presented as full-source coverage.

## 5. Storage, publication and undo

Use fresh, isolated ordinary SQLite tables in the experiment. Do not alter vec0
shadow tables or adopt an application database. The target storage roles are:

| Proposed table | Key/content | Authority and history |
|---|---|---|
| `li_profiles` | Profile digest, immutable profile metadata | Retained representation contract |
| `li_artifacts` | Integer rowid; unique artifact digest; profile, role/input digest, token count, dimensions, matrix BLOB | Immutable retained input; stream with SQLite BLOB APIs where useful |
| `li_document_refs` | `(corpus, document_id, profile)` -> artifact digest | Small canonical reference, eligible for managed history capture |
| `li_index_state` | Same identity -> current source epoch and artifact digest | Derived publication binding; recomputed on source changes and undo |

Fixture source tables supply body, live state and monotonic source epoch. They
are experiment-owned, not a new mandatory application data model. Production
bindings must name the owning source relation and epoch explicitly.

Artifact digests cover profile, shape and payload. Define one host-side SHA256
encoding: domain tag `sqlite-vec/li-matrix/v1` followed by a zero byte, the 32-byte
profile digest, one role byte (`0` document, `1` query), the 32-byte input digest,
little-endian uint32 token count/dimension, then matrix bytes.
Profile digests use a versioned JSON object with sorted keys, compact separators,
UTF-8, no duplicate keys, and strings/integers/Booleans only; decimal parameters
are canonical strings. Freeze test vectors before persisting real artifacts.
Hashes identify bytes; they do not prove a model actually produced them.

Publication protocol for later managed-writer integration:

1. Read source identity/epoch and compute token embeddings outside the write lock.
2. Validate and stage immutable artifact bytes through a bounded, explicit host
   ingestion path. Staging may commit unreferenced artifacts; it never makes
   them eligible search results. Duplicate digests must verify matching bytes.
3. Under `BEGIN IMMEDIATE`, recheck source epoch, profile and liveness. Publish
   canonical references, derived epoch bindings, applicable source/FTS/dense
   updates, history links and retry receipt atomically. A stale preparation fails.
4. Undo restores the saved canonical artifact reference and source values, then
   allocates a newer source epoch and rebuilds derived bindings. Never invert
   epoch metadata or copy index shadow BLOBs. Claim/undo-claim preserves source
   epoch and performs no artifact/index rewrite.
5. Retain artifacts reachable from current references **or retained history**.
   Initial experiments perform no artifact GC. Production reclamation needs
   explicit history-reference indexing, snapshot/backup retention and recovery
   tests before any deletion is enabled.

An epoch change with identical late-model input/profile may reuse verified artifact
bytes and refresh the derived binding atomically. Do not rerun a model solely
because a pooled dense vector changed; do not stamp stale text artifacts with a
new epoch without verifying the exact input digest.

Large matrices stay outside Session changeset payloads; history captures small
references, not whole token matrices. This is a proposed extension, not supported
by the current version-2 writer. It requires a fresh schema and explicit managed
ingestion boundary; existing limits and fixtures remain unchanged.

A source revision without a matching ready artifact cannot use an older matrix.
Late-interaction requests refuse it. Ordinary baseline search remains available
as its existing explicit operation. Do not hide a fallback or omit stale candidates
while labelling the reranked set complete. Derived bindings must not constrain
Session's temporary canonical state; reuse the existing post-inverse validation
pattern instead of introducing the already-rejected transient foreign-key shape.

## 6. Resource and cancellation contract

Process one document at a time; never materialize the query×document similarity
matrix or all candidate token matrices. Extra scorer scratch is `O(query_tokens)`
or smaller. Borrow input bytes where safe, decode little-endian float values
without unaligned pointer casts, and avoid a resident model/runtime in C.
SQLite may materialize a document BLOB and retain pager memory: borrowed scorer
inputs do not establish end-to-end zero-copy or a process RSS limit.

Provisional **experiment admission ceilings**, not measured production defaults:

| Quantity | Ceiling |
|---|---:|
| Distinct candidates in one corpus | 100 |
| Query rows / document rows | 128 / 4,096 |
| Dimensions | 4,096 |
| Query bytes / one document's bytes | 1 MiB / 8 MiB |
| Aggregate candidate matrix bytes | 32 MiB |
| Charged work units per request | 134,217,728 |

All ceilings apply together. A valid model dimension alone does not imply its
largest matrix or a 100-document request fits the byte/work ceilings. Reject
excess work; do not shorten the candidate set, discard tokens or return partial
scores to fit. Model truncation is an explicit ingestion/profile decision.

Charge one unit per component during matrix validation, one per component product
during dot scoring (including operand decoding), one per pairwise maximum
comparison and one per outer score accumulation. Admission computes
checked cost `m*d + sum_D(m*d + n*d + m*n*d + m*n + m)` before scoring: the first
term admits the query, and every standalone scalar call validates its query input
again. Native validation and execution consume the same accounting; overflow
rejects before effects. SQL VM work, digest hashing and I/O are separate costs;
retain byte/VM admission and report those costs rather than claiming the numerical
budget bounds them.
Pass only the remaining budget to successive documents, not a fresh full budget
per scalar call. Record admitted and consumed work/bytes in the experiment.

Poll cancellation in validation and scoring loops at intervals of at most 1,024
charged units and at document boundaries. Reuse the graph-witness interruption
compatibility approach; qualify bounded native cancellation on SQLite 3.41+.
Document older-host capability refusal without raising the whole extension's
minimum version silently. A progress handler bounds separate SQLite VM work;
neither it nor an outer LIMIT interrupts an uninstrumented native loop.
All budget/cancellation failures discard partial results and close the read
transaction. Wall-clock latency and RSS are measured outcomes, not guarantees
derived from these logical work/payload budgets.

## 7. Proposed SQL and host interfaces

Only after the experiment gate, propose this standalone direct-only SQL function:

`vec_maxsim(query_matrix_blob, document_matrix_blob, dimensions, work_limit)`

This is a proposed name/signature, not an existing SQL API. It returns a float64
score under section 3, requires non-NULL inputs, and accepts no table names,
models, arbitrary SQL or filesystem paths. The host checks profile/source identity.
The kernel validates shape, finite values, row normalization and its local budget.
No JSON overload, quantizer, GPU, cache, batch API or metric switch in v1.
Matrix arguments must have SQLite BLOB type; dimension and work arguments must
have INTEGER type in their admitted positive ranges. Do not coerce TEXT, REAL or
NULL into a valid request. Metadata admission uses BLOB lengths before the host
loads candidate payloads, then streams one candidate through scoring at a time.

Error classes: malformed/type/shape/nonfinite/normalization -> `SQLITE_MISMATCH`;
payload size -> `SQLITE_TOOBIG`; exhausted work -> `SQLITE_FULL`; cancellation ->
`SQLITE_INTERRUPT`; allocation failure -> `SQLITE_NOMEM`. Prefix diagnostics with
`vec_maxsim:` so future adapter mapping does not misclassify unrelated SQL errors.
Never return SQL NULL, a partial score or zero for a rejected representation.

The experiment runner consumes explicit precomputed inputs. It must not invoke
an embedding tool/model implicitly. For optional later CLI/MCP integration:

- A named opt-in `rerank` operation uses the existing shared handler boundary.
  Trusted corpus configuration binds artifact tables/profile and source epochs.
- Candidate generation plus artifact admission/scoring uses **one** read
  transaction. Separate existing `search` and `rerank` calls do not promise a
  shared snapshot. External candidate IDs are revalidated against current scope
  and can only claim reranking of that explicitly supplied set.
- Prefer bounded query-artifact references to large JSON matrices. The existing
  65,536-byte input envelope remains intact; do not add silent base64/inline
  payload exceptions. Query preparation is an explicit producer operation.
- Preserve the current external-ID cap of 64; any 100-candidate internal window
  belongs to the new operation's declared limits. The existing 64 MiB database
  cap will exclude many real token corpora: 5,183 documents × 256 tokens × 128
  dimensions × 4 bytes is about 648 MiB before SQLite overhead. L4 needs measured
  corpus/resource admission review; it must not simply remove that cap. L0/L1
  run independently of this fixture-scale adapter admission.
- Return score kind/direction/profile, source epoch, candidate count, first-stage
  rank and stage provenance, token coverage/truncation, work used and bounded
  candidate-set completeness. No claim of exhaustive corpus recall.
- Add `describe` capability and structured missing/stale/model/budget errors.
  Preserve default operations, JSON/JSONL envelopes, read-only authorization,
  MCP stdout and cancellation isolation for queued/subsequent calls.

Wire schema versioning and query-artifact admission require a separate interface
review before that phase; this proposal does not preauthorize public write tools.

## 8. Implementation slices and ownership

| Slice | Proposed files/owners | Exit condition |
|---|---|---|
| L0: reference | This archive: `oracle.py`, tiny fixtures, profile/digest vectors, `test_oracle.py` | Hand-computed and metamorphic score/shape/budget cases pass; negative controls fail |
| L1: measured text experiment | Archive: explicit preparation records, SQLite fixture, `benchmark.py`, frozen manifests/receipts/report | Real relevance and resource gate below; no C/adapter/writer changes yet |
| L2: native scorer, only if useful | `sqlite-vec-late-interaction.c`, registration/include in `sqlite-vec.c`, `reference.yaml`, `tests/test-late-interaction.py`, C unit owner, loadable inventory/build checks | Oracle equivalence, failure/cancellation/budget tests, sanitizers and measurable benefit over reference |
| L3: managed artifacts, if selected | Existing writer plus focused ordinary-table/artifact tests and README | Atomic publication, bounded history, undo/retry/crash/snapshot guarantees including artifact retention |
| L4: opt-in host transport | Search-lab shared handler, CLI/MCP schemas, config/capability/docs and parity tests | One-snapshot end-to-end parity and all existing read-only/cancellation checks pass |

Use existing test owners instead of duplicating process/transport fixtures. The
modular C include/build pattern is the candidate integration point; verify actual
registration/build rules when implementing. No core function is added in L0/L1.
Update `ARCHITECTURE.md` when a slice changes current capabilities or contracts,
not to describe planned behavior as shipped. Add a single roadmap link only when
this research is selected for execution; do not reorder current safety work here.

## 9. Verification and experiment gate

Numerical oracle: independent scalar Python float64 loops plus a small relational
MAX-then-SUM fixture. Hand cases cover orthogonal and all-negative rows, ties,
repeated query tokens, duplicate/permuted document rows, masking and signed IDs.
For optimized/native comparisons require `abs(error) <= 1e-6 + 1e-6*abs(reference)`;
compare ordered IDs exactly on fixtures whose score gaps exceed that tolerance.
Near ties retain actual score differences; the ID rule applies only to equal
computed scores. Do not round scores to manufacture ranking parity.

Negative controls must make the intended assertion fail: initialize maxima at
zero; omit an admitted query/document row; always return first-stage order; reset
budget per candidate; use stale/model-mismatched artifacts; omit a derived epoch
update; or inject an orphan reference. Track expected versus observed score,
execution path, scored candidates, pair/component counts and matrix bytes read.
An answer-only comparison cannot prove that reranking actually ran.

Exercise lengths/overflow, normalization, all ceilings and one-unit boundary
failures, missing/empty candidates, corrupted artifacts, cancellation while native
work is active, queued cancellation and retry after failure. Use deterministic
execution coordination, not a sleep that merely hopes to interrupt the kernel.
L3 additionally covers old/new reader snapshots, model switches, content races,
claim effects, artifact staging/publication crashes, strict multi-row undo and
backup/restore with retained history. No garbage collector is accepted by age alone.

Real-data experiment: reuse the existing SciFact corpus/ID provenance where
applicable, but explicitly pin corpus text, qrels, preprocessing and embeddings.
Choose disjoint tuning and evaluation queries before tuning; do not rename earlier
used queries as a new blind holdout. Synthetic fixtures prove arithmetic, not
retrieval quality. Keep first-stage candidates identical for baseline/reranked
comparisons and separately report relevant-document coverage of that candidate set.

Compare BM25, dense and RRF baselines against each plus late reranking. Report
NDCG@10, MRR@10, labelled candidate recall, p50/p95 total and scoring latency,
peak RSS, SQLite API/blob bytes, artifact/DB/WAL bytes, ingestion cost and churn.
Keep warm/cold diagnostics, model inference and scoring costs distinct; include
inference in the end-to-end result. Use paired runs and bootstrap intervals over
queries, with retained seeds, raw observations and corruption checks.

Promotion requires a predeclared target-workload budget recorded **before** the
held-out run. Select and freeze the strongest baseline on the tuning split. The
initial relevance target is +0.02 absolute NDCG@10 over that baseline, with a
paired 95% bootstrap interval excluding zero.
The hardware-specific p95/RSS/disk/inference caps remain uncalibrated until L1
preparation; an unset cap blocks promotion, not the experiment. Record exact caps
before inspecting evaluation outcomes. If model-compatible input/labels cannot
be obtained, mark relevance unverified rather than substituting synthetic wins.

Add native C only when L1 demonstrates useful relevance within those budgets and
profiling identifies reference scoring as a material cost. If gains fail, retain
the negative result and stop. L2 must preserve semantics and reduce measured cost;
its presence alone is not a quality result. Never change thresholds after a failed
held-out run to label it passed.

## 10. Deployment, rollback and unresolved decisions

L0/L1 use new disposable fixtures and explicit offline producers. No installs,
model downloads, migrations, existing-data adoption, push or CI changes occur as
part of authoring this plan. Recheck worktree ownership before implementation.
First implementation needs no new runtime dependency beyond the selected explicit
model producer; review that producer separately before executing/installing it.

Native rollout is additive and optional, with default/IVF/no-HNSW/amalgamation/
static/Rust-consumer checks. Disable the opt-in reranker to return to the existing
explicit search operation; do not automatically rerun failed late requests as
baseline successes. Preserve artifact/history data when reverting a binary.
Inspect integrity, revision/profile/digest bindings and exact structured failure
first when diagnosing a rejected result; do not repair by bypassing freshness.

Decisions to close in L1: exact checkpoint/license/runtime, corpus query split,
source representation/chunk coverage, hardware-specific resource caps and whether
the measured relevance gain warrants L2. These do not block the independent L0
reference contract. Token compression, approximate token retrieval, cross-encoders
and multimodal models require their own score/quality/resource comparisons.

## Sources and evidence limits

- [OpenSearch late-interaction article](https://opensearch.org/blog/boost-search-relevance-with-late-interaction-models/):
  selected scoring/two-stage/storage sections reviewed. It motivates the experiment;
  its example is not evidence of a gain on this fork's data.
- [ColBERT paper](https://arxiv.org/abs/2004.12832): abstract reviewed for independent
  query/document encoding and offline document representations; no reproduction
  or full-paper benchmark audit is claimed.
- [Official ColBERT implementation](https://github.com/stanford-futuredata/ColBERT):
  entry point for checkpoint-specific encoding/masking/score qualification, which
  must be inspected at the pinned revision during L1.
- Current local owners listed in section 2, plus
  [writer acceptance](../2026-11-08-writer-sync/ACCEPTANCE.md) and
  [current roadmap](../../../TODO.md). This document contains proposed contracts
  and acceptance targets, not measured late-interaction results.
