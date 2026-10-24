"""Graph + vector retrieval, end to end.

The graph path and the vector path compose through the one shared
filtered-KNN contract (see tests/test-filtered-knn.py): `graph_bfs(...)`
produces node ids, `rowid in (select node from graph_bfs(...))` feeds them
to vec0 as the `KNN_ROWID_IN` filter, and vec0 either ranks the named rows
exactly (filter of at most VEC0_KNN_EXACT_FILTER_MAX = 4096 rows) or applies
the filter while the index is traversed (larger filters).

Selective filters can legitimately return fewer than k rows; no test here
assumes len(result) == k except where the exact-ranking path guarantees it
(the whole filtered set is ranked, so it returns min(k, |filter|) rows).
"""
import math
import random
import sqlite3
from collections import deque

import pytest

from helpers import _f32

DIM = 8
EXACT_FILTER_MAX = 4096  # VEC0_KNN_EXACT_FILTER_MAX in sqlite-vec.c


def connect():
    db = sqlite3.connect(":memory:", isolation_level=None)
    db.enable_load_extension(True)
    db.load_extension("dist/vec0")
    db.enable_load_extension(False)
    return db


def make_vecs(n, seed=42):
    rng = random.Random(seed)
    return {i: [rng.random() * 10 for _ in range(DIM)] for i in range(1, n + 1)}


def ring_edges(n):
    """Each node points at the next two nodes on a ring: i -> i+1, i -> i+2."""
    edges = []
    for i in range(1, n + 1):
        edges.append((i, i % n + 1))
        edges.append((i, (i + 1) % n + 1))
    return edges


def ref_bfs(edges, start, max_depth=None):
    adj = {}
    for s, d in edges:
        adj.setdefault(s, []).append(d)
    seen = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        if max_depth is not None and seen[u] >= max_depth:
            continue
        for v in adj.get(u, []):
            if v not in seen:
                seen[v] = seen[u] + 1
                q.append(v)
    return seen


def brute_knn(vecs, query, allowed, k):
    """Exact ranking over `allowed` rowids, nearest first."""
    ranked = sorted(
        allowed,
        key=lambda i: (sum((a - b) ** 2 for a, b in zip(vecs[i], query)), i),
    )
    return ranked[:k]


def build(db, vecs, edges, hnsw):
    db.execute("create table edges(src integer, dst integer)")
    db.executemany("insert into edges values (?, ?)", edges)
    opts = " indexed by hnsw(m=8, ef_construction=64)" if hnsw else ""
    db.execute(f"create virtual table docs using vec0(v float[{DIM}]{opts})")
    db.executemany(
        "insert into docs(rowid, v) values (?, ?)",
        [(i, _f32(vec)) for i, vec in vecs.items()],
    )


def bfs_nodes(db, start, max_depth):
    return {
        r[0]
        for r in db.execute(
            "select node from graph_bfs('edges', 'src', 'dst', ?, ?)",
            [start, max_depth],
        )
    }


def filtered_knn(db, query, start, max_depth, k):
    return db.execute(
        "select rowid, distance from docs where v match ? and k = ? "
        "and rowid in (select node from graph_bfs('edges', 'src', 'dst', ?, ?))",
        [_f32(query), k, start, max_depth],
    ).fetchall()


@pytest.mark.parametrize("hnsw", [False, True], ids=["flat", "hnsw"])
def test_khop_neighborhood_as_vector_filter(hnsw):
    """k-hop neighborhood as a vector filter, on flat and HNSW vec0 tables.

    The neighborhood is small (<= 4096 rows), so the shared path ranks every
    named row by its exact distance: the result must equal a brute-force
    exact ranking over the same neighborhood computed in Python.
    """
    vecs = make_vecs(60)
    edges = ring_edges(60)
    db = connect()
    build(db, vecs, edges, hnsw)

    neighborhood = bfs_nodes(db, 1, 3)
    assert neighborhood == set(ref_bfs(edges, 1, 3))
    assert 0 < len(neighborhood) <= EXACT_FILTER_MAX

    rng = random.Random(99)
    query = [rng.random() * 10 for _ in range(DIM)]
    rows = filtered_knn(db, query, 1, 3, k=10)
    assert [r[0] for r in rows] == brute_knn(vecs, query, neighborhood, 10)
    # Exact ranking returns the whole neighborhood when k exceeds it, with
    # distances equal to the brute-force Euclidean distances.
    assert len(rows) == len(neighborhood)
    for rowid, distance in rows:
        exact = math.sqrt(sum((a - b) ** 2 for a, b in zip(vecs[rowid], query)))
        assert distance == pytest.approx(exact, rel=1e-5)


@pytest.mark.parametrize("hnsw", [False, True], ids=["flat", "hnsw"])
def test_vector_seed_then_khop_expansion(hnsw):
    """Vector seed, then k-hop expansion: KNN results feed graph_bfs starts
    and the expanded node set equals the Python reference expansion."""
    vecs = make_vecs(60)
    edges = ring_edges(60)
    db = connect()
    build(db, vecs, edges, hnsw)

    rng = random.Random(7)
    query = [rng.random() * 10 for _ in range(DIM)]
    seeds = [
        r[0]
        for r in db.execute(
            "select rowid from docs where v match ? and k = 3", [_f32(query)]
        )
    ]
    assert seeds == brute_knn(vecs, query, set(vecs), 3)

    expanded = set()
    for seed in seeds:
        expanded |= bfs_nodes(db, seed, 1)
    want = set()
    for seed in seeds:
        want |= set(ref_bfs(edges, seed, 1))
    assert expanded == want
    assert set(seeds) <= expanded


def test_large_neighborhood_uses_filtered_traversal():
    """A neighborhood above VEC0_KNN_EXACT_FILTER_MAX (4096) on HNSW.

    The shared path applies the filter during HNSW traversal instead of
    ranking exactly. The documented caveat (tests/test-filtered-knn.py) is
    that such a search can legitimately return fewer than k rows, so this
    test does not assert len == k; it asserts what must still hold against
    the brute-force reference: every returned row is in the neighborhood,
    distances are the exact brute-force distances of the returned rows, and
    they come back nearest-first.
    """
    n = 6000
    vecs = make_vecs(n, seed=1234)
    # A chain 1 -> 2 -> ... -> n: BFS from 1 reaches the first 5000 nodes.
    edges = [(i, i + 1) for i in range(1, n)]
    db = connect()
    build(db, vecs, edges, hnsw=True)

    neighborhood = bfs_nodes(db, 1, 4999)
    assert neighborhood == set(range(1, 5001))
    assert len(neighborhood) > EXACT_FILTER_MAX

    # The query is a vector outside the neighborhood: its exact nearest
    # rows overall are excluded, so only in-neighborhood rows may return.
    query = vecs[5500]
    assert brute_knn(vecs, query, set(vecs), 1) == [5500]
    rows = filtered_knn(db, query, 1, 4999, k=10)

    assert 1 <= len(rows) <= 10
    assert all(rowid in neighborhood for rowid, _ in rows)
    distances = [d for _, d in rows]
    assert distances == sorted(distances)
    for rowid, distance in rows:
        exact = math.sqrt(sum((a - b) ** 2 for a, b in zip(vecs[rowid], query)))
        assert distance == pytest.approx(exact, rel=1e-5)
