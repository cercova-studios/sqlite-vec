# Graph semantics and conformance fixtures

These fixtures specify existing native graph behavior and the search adapter's
catalog/scope rules. Run them before changing graph execution or porting the host
adapter. They establish G0's baseline; they add no GraphIR request language or binder.

```sh
uv run --project examples/search-lab --group dev pytest -q \
  examples/search-lab/test_graph_semantics.py
```

The tests require a built `dist/vec0.dylib` or `dist/vec0.so`, as described in the
[adapter setup](README.md#run-the-fixture). The
[JSON fixtures](graph-fixtures.json) contain graph data and handwritten expected
rows. The test-only [oracle](graph_oracle.py) computes shortest distances with
Floyd-Warshall all-pairs relaxation. It imports neither SQLite nor adapter/native
traversal code. Its cubic cost is appropriate only for these tiny test graphs.

## Identity and multiplicity

A fixture node identity is `(graph, node_type, signed_int64_id)`. The node type
identifies an ID domain, not a collection of property labels. `Function(1)` and
`File(1)` in the same graph are distinct; so are `Function(1)` in two graphs.
IDs include zero, negative values and both signed 64-bit endpoints. JSON consumers
must preserve integer precision. A nullable property does not make the node absent.

A fixture edge identity is `(graph, edge_id)`. Parallel edges have separate edge
IDs even when their endpoints agree. Edge IDs are unique within one fixture graph;
the same integer may identify edges in different graphs. These fixture IDs are
ordinary SQL data. Native traversal receives endpoint pairs and returns no edge ID.

Native `graph_bfs` has no node catalog or type binder. The selected table/view and
source/destination columns supply the traversal domain. A source must not combine
independent integer ID domains unless the host first supplies a collision-free
mapping. The adapter's current ID domain is `(configured corpus, document ID)`;
it does not accept a `node_type` parameter.

Reachability is a **distinct node set** with one minimum distance per node.
Cycles, self-loops, parallel edges and alternate diamond routes do not duplicate
nodes or count paths. Ordinary SQL edge bindings retain bag multiplicity. The
diamond fixture has three outgoing edge rows from node 1, but only two distinct
one-hop neighbors. Reusing one node variable means equality of its full identity,
not equality of integers drawn from unrelated domains.

## Reachability and shortest paths

| Condition | Native helper | Adapter `traverse` |
|---|---|---|
| Direction | Source → destination; swap configured columns or use a reverse/union view for another direction | Outgoing edges from the configured source |
| Start node | Always included at depth zero after source validation, even if absent from every edge | Included only if the document exists and is eligible |
| Existing isolated node | Indistinguishable from an absent anchor; both return the anchor | Returns the eligible node at depth zero |
| Absent/ineligible anchor | Native helper has no eligibility/catalog check | Listed in `missing_ids`; contributes no reachable nodes |
| Maximum depth | Inclusive nonnegative integer; omitted or NULL means unbounded | Inclusive integer 0–8; default 1 |
| Multiple seeds | One anchor per native invocation | Union of admitted seed results, with minimum distance from any seed |
| Ordering | Nondecreasing BFS depth; no promised same-depth tie order | Depth ascending, then node ID ascending |
| Output limit | SQL may limit emitted rows after native materialization | Limits the sorted result after computing total reachable nodes |

Tests compare canonical native rows with an explicit `ORDER BY depth,node`.
They do not freeze incidental insertion or adjacency order. Reversing edge
insertion order, duplicating edges and adding self-loops preserve minimum distances.

`graph_shortest_path` returns one unweighted shortest **node sequence**, with
consecutive steps starting at zero. It returns no rows when different endpoints
are disconnected. Equal endpoints return a one-node zero-step sequence even if
the ID is absent from the edge source. Tests accept any valid shortest sequence
when ties exist. They check endpoints, consecutive edges, length and step numbering;
they do not invent relationship witnesses or an all-shortest-path contract.

The maximum depth bounds minimum-distance reachability. It is not an exact-length
path predicate. In the shortcut fixture, `1 → 2 → 3` and `1 → 3` both exist.
Node 3 has minimum distance one, so filtering BFS rows with `depth=2` omits it,
although a two-edge binding exists. A future lower-hop bound cannot silently mean
existence of a path in an interval.

Edge labels are ordinary SQL attributes. A selected `CALLS` view filters the
admitted edge relation **before** traversal. Every hop then uses that relation.
This does not implement a sequence such as `CALLS/FLOWS_TO`, an automaton, weights,
trail semantics or path enumeration.

## NULL and optional bindings

Native graph inputs are SQLite integer endpoints. Source validation runs before
traversal, including at maximum depth zero. A NULL source skips that row without
examining its destination. With an integer source, a NULL destination skips the
row. An invalid non-NULL source still fails when its destination is NULL. Other
non-NULL non-integer endpoints fail with an explicit SQLite error. Missing sources
or columns and invalid anchor/depth types also fail; they are not empty results.

SQL optional bindings use ordinary `LEFT JOIN` behavior. An isolated catalog node
can have one output binding with NULL edge and destination fields. `COUNT(*)` then
counts one binding while `COUNT(edge_id)` counts zero edges. A nullable node name
uses `IS NULL`; `name = NULL` remains unknown. An absent catalog ID contributes no
node row. These are verified SQL baselines, not additional adapter operations.

## Scope and transaction visibility

The host's approved edge relation defines the traversal subgraph. Filtering
reachable output after traversing a larger graph can admit paths through excluded
intermediaries. The scope fixture places an eligible isolated node behind an
ineligible bridge: it is reachable in the raw graph and absent from scoped
reachability. The existing adapter rejects a configured source containing any
ineligible endpoint. Its fixed tenant, validity and invalidation rules remain
[the adapter contract](README.md#configure-a-corpus), not an authorization service.

Native traversal sees same-connection writes. The fixtures warm the cache, insert
an edge in a transaction, delete routes inside a savepoint, roll back that
savepoint, roll back the transaction, and finally commit an insertion. Every
observed state is checked against the independent oracle. Cache reuse must never
make a rolled-back edge visible or hide an uncommitted edge.

Each adapter operation retains its existing explicit read transaction. Separate
operations can observe different commits. Neither the fixture identity nor a
persistent connection establishes a cross-call snapshot. Work, memory and
cancellation guarantees remain those documented in the adapter; an output limit
does not add a native execution budget.

## Fixture coverage

| Fixture/check | Distinguishing expectation |
|---|---|
| `diamond` | One row per node despite alternate routes and parallel edges; isolated and absent anchors differ only after catalog admission |
| `cycles` | Termination with minimum distances; repeated-variable equality selects self-loops |
| `shortcut` | Minimum distance differs from exact-length binding existence |
| `integer_ids` | Zero and full signed 64-bit IDs survive traversal |
| `null_endpoints` | Skipped NULL endpoint rows do not create neighbors |
| `empty` | Native anchor remains at depth zero without edges |
| Identity/SQL tests | Graph/type domains, edge identity, bag counts, optional NULL bindings and label-filtered sources |
| Scope/adapter test | Independent eligibility oracle, multiple seeds, missing IDs and paging/completeness |
| Transaction test | Own writes, savepoint rollback, full rollback and commit |
| Seeded graphs | 24 tiny graphs, all selected anchors/depths, reversed insertion order, duplicate edges and self-loops |

The tests preserve the existing C graph code and A0 handlers. G0 is a contract and
conformance baseline; a validated reachability compiler belongs to G1, and native
resource accounting belongs to G2. Detailed validation belongs in the
[G0 receipt](experiments/2026-10-26-g0/VALIDATION.md).
