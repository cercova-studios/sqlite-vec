# P4/V1 write and history contract

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../README.md).

Design for the next experimental slice, not a shipped API. The existing six
read-only operations in `examples/search-lab/src/lib.rs` and their CLI/MCP
envelopes remain unchanged. No existing database is enrolled or migrated here.

## Target and first runnable slice

Keep canonical application data in ordinary SQLite tables. A managed request
atomically validates revisions, mutates records, maintains derived indexes and
persists attribution/history plus a retry receipt. Use SQLite transactions and
its single-writer serialization; neither a storage-engine fork nor another
database process is needed.

The reference design supports explicitly registered tables with immutable,
non-NULL primary keys, including typed composite keys. The first implementation
admits single `INTEGER PRIMARY KEY` tables in fresh fixtures. This is an explicit
coverage restriction, not an argument that other keys are unnecessary. It avoids
shipping an unverified generic key codec. Application-specific fields and claim
rules stay in host handlers; records do not become a mandatory generic JSON store.

## Boundary and ownership

Proposed Rust boundary: `Writer::apply(validated_batch, host_identity, cancellation)`
returns a durable receipt. `Writer` owns a separate write connection. `Adapter`
keeps its current read-only connection, flags, authorizer and per-operation read
transactions. Do not relax those protections to share a connection with writes.
Reuse the existing binary/transports when wiring the writer; do not duplicate a
second retrieval engine or add write code to the SQLite-only C core.

The trusted host opts in at startup and fixes the database, project identity,
actor identity, table registry and permitted named handlers. Clients supply a
request ID, handler arguments and expected row revisions, never SQL, filesystem
paths, extension names, table mappings or an authoritative actor. Corpus tenant
eligibility remains separate from authorization. Writer discovery is enabled only
for an opted-in host; default tool discovery and config behavior stay unchanged.

Keep the existing 65,536-byte request and 262,144-byte result bounds. Initially
admit at most 64 distinct table/key mutations per batch, with indexed primary-key
lookups and bounded old/new row payloads. Row/capture byte bounds need a concrete
implementation and tests before exposure; a small request can address a huge row.
Never advertise a process RSS bound from these admission limits or Session's
post-allocation memory counter.

## Identities and revisions

The first fixture needs ordinary protected metadata tables with these roles:

| Record | Identity and invariant |
|---|---|
| Registered table | Stable integer table ID, table name, primary-key/schema signature; never infer an identity from a mutable name alone |
| Request | `(project, request_id)` unique; trusted actor, canonical validated request, receipt and commit sequence stored together |
| Row version | `(table_id, integer_key)` unique; current revision and live/deleted state survive hard deletes |
| Change batch | Request/commit identity, schema identity and typed old/new capture payload |

Allocate a monotonically increasing commit sequence under the write transaction.
Every row changed by that request receives that sequence in the authoritative row
version table. Ordinary application tables need no duplicate revision column.
Reject sequence overflow. A failed transaction publishes no valid sequence.

Creation requires an unused identity; hard deletion retains a tombstone. Normal
create cannot reuse a deleted key. An explicit undo may restore it with a newer
revision. This avoids delete/recreate and undo ABA bugs, where a stale revision
appears current again. The benchmark's in-row `rev` is a counterexample fixture,
not the proposed production revision storage.

Retain request keys and identity high-water marks. Forgetting either can weaken
deduplication or permit ID reuse. History pruning, idempotency expiry and tombstone
compaction require an explicit retention contract before implementation.

## Request state machine

1. Validate the request shape, host handler, scope and bounds before mutations.
   Canonicalize the validated operation and arguments, including expected revisions
   and trusted identity; compare canonical bytes rather than requiring a hash service.
2. Acquire `BEGIN IMMEDIATE` with a bounded, cancellable busy policy. Check the
   request ledger under that lock. A matching key/content returns its stored receipt;
   a key with different content or actor fails without applying anything.
3. Read admitted old rows and row versions on this connection. Validate schema,
   all expected revisions, payload budgets and domain constraints before mutation.
   Register capture for the explicit canonical-table allowlist only.
4. Apply the host's prepared, parameterized mutations. Maintain row versions and
   derived indexes; capture typed net changes. Store actor/request/schema metadata,
   capture payload and the immutable receipt within the same transaction.
5. Commit once. Return success only after commit succeeds. On failure or cancellation
   before commit, roll back and discard capture state. Never reuse a Session object
   across failed or unrelated requests.

Retries use the same request ID. Concurrent identical requests serialize to one
commit and one receipt. A stale claim/update conflicts; it is not silently retried
with a freshly read revision. A lost response after commit is resolved through the
ledger, not a new request ID. Cancellation after successful commit cannot turn the
receipt into an aborted write. Ambiguous I/O/commit outcomes require discarding the
connection and reporting an unknown outcome; reopen and resolve the original key.

The response can identify a replay, but the stored committed result stays immutable.
Retain the existing result/error envelope conventions. Before adding public tools,
specify versioned request schemas and stable error codes for disabled writes,
invalid input, reused keys, revision/domain conflict, busy, cancellation, schema
drift, resource limits and unknown outcome. Those names are not exposed today.

## Capture decision

Prefer a fresh per-request SQLite Session object as the first capture prototype,
paired with an explicit SQL request/revision ledger. It already preserves SQLite
value types and computes net changes, avoiding a new generic trigger/serialization
engine. Enable Session/preupdate only in that experimental writer host; leave the
C core's SQLite-only dependency boundary intact. Attach explicit canonical tables,
never all tables or index shadow tables. One component must own preupdate-hook use
on that write connection; check host-library compatibility before integrating.

The comparison in [README](README.md) is evidence for this prototype, not proof
that Session is a complete history engine or the fastest complete writer. Session
has no actor/request ledger, row-version policy or direct SQL history index. The
next prototype must add and measure those pieces. Decode a changeset to produce
an indexed changed-row catalog if history lookups need it; include that cost in
the next comparison. The SQL log remains a portable alternative, especially when
host Session support or SQL-queryable history matters more than capture overhead.

Do not migrate existing projects or enable Session in the default bundle based
only on this experiment. No capture mechanism records unmanaged external writers
automatically across every connection. All managed writers must use the boundary;
schema/version drift and unsupported writer connections must fail admission rather
than imply complete audit coverage. A host with arbitrary SQL access can bypass
application policy; the history mechanism is not an authentication boundary.

## Undo is a new validated request

Read the target request's typed changes and recorded after-revisions. Acquire the
write lock and require each affected row/tombstone to still have that revision.
If any row has an intervening write, reject the entire undo. Start with conservative
row-level conflicts; do not merge disjoint fields implicitly.

Apply inverse application values through the same handlers/constraints, allocate
a new commit sequence, update row versions, rebuild affected derived data and append
new history/attribution linked to the target. Never apply an unfiltered raw inverse
to the revision ledger. The native probe demonstrates that raw inversion rewinds
an in-row revision. No conflict callback may silently `OMIT` or `REPLACE` a conflict.
Foreign keys, unique constraints, exclusive claims and domain graph constraints
are obligations of the complete transaction, not facts guaranteed by cell merging.

## Derived retrieval state

Canonical rows and model identity are authoritative. SQLite FTS/vec0 structures are
derived and never merged as opaque shadow blobs. On a managed write, update them
on the same connection/transaction, or mark the revision unavailable for retrieval
until a verified rebuild. Never serve a stale index as current.

Compute embeddings outside the write lock. Before applying them, validate the
expected source revision, source-content identity and model/dimension/metric identity
under the lock. Expensive embedding work must not hold SQLite's writer slot.
Initial task-only prototypes may omit retrieval synchronization only when that
capability is explicitly unavailable. Lance/provider transactions require their own
integration/rollback checks before admission; this contract does not certify them.

## Next implementation and acceptance

Implement one fresh-fixture Rust transaction path, not public write tools yet.
Its domain journey is two agents racing to claim one ordinary task row, followed
by a retried request, conflict-aware undo and history lookup. Check:

1. Exactly one stale-revision claim wins; identical retries create no second write.
   A reused key with changed arguments fails. Deletion/restoration never reuses a revision.
2. Statement/savepoint/full rollback, capture failure, cancellation and process exit
   leave data, versions, receipt and history either all committed or all absent.
3. Undo preserves later writes, remains atomic across multiple rows and produces a
   newer revision. Verify BLOB/NULL/type preservation and constraint failures.
4. Existing CLI/MCP/G0 parity and read-only rejection remain unchanged. Measure the
   full request/history pipeline with its actual memory and byte admission rules.
5. Add source/FTS/vector synchronization and model-staleness tests before enabling
   writes against retrieval corpora. Existing-data enrollment needs migration review.

Default-branch capabilities stay documented as shipped; this design belongs in
the development archive until a runnable implementation establishes its guarantees.
