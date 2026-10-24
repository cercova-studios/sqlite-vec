"""Shared filtered-KNN contract: `rowid in (...)` / `id in (...)` on vec0.

One policy for every index (flat, rescore, IVF, DiskANN, HNSW): a filter
naming at most VEC0_KNN_EXACT_FILTER_MAX (4096) rows ranks each named row by
its exact distance; a larger filter is applied while the index is searched.
The filter is the same VTAB `IN` argument the graph path feeds with
`rowid in (select node from graph_bfs(...))`; there is no second mechanism.
"""
import random
import sqlite3

import pytest

from helpers import _f32


def _has_ivf(db):
    flags = db.execute("SELECT vec_debug()").fetchone()[0].split("Build flags:")[-1]
    return "ivf" in flags


INDEX_OPTS = {
    "flat": "",
    "hnsw": " indexed by hnsw(m=8, ef_construction=64)",
    "diskann": " indexed by diskann(neighbor_quantizer=binary, n_neighbors=8)",
    "rescore": " indexed by rescore(quantizer=int8, oversample=8)",
    "ivf": " indexed by ivf(nlist=4, nprobe=4)",
}


def _vec(i):
    return _f32([float(i), float(i % 3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _build(db, index, text_pk=False):
    if index == "ivf" and not _has_ivf(db):
        pytest.skip("IVF not enabled (compile with -DSQLITE_VEC_EXPERIMENTAL_IVF_ENABLE=1)")
    opts = INDEX_OPTS[index]
    if text_pk:
        db.execute(f"create virtual table t using vec0(id text primary key, v float[8]{opts})")
        db.execute("create virtual table ref using vec0(id text primary key, v float[8])")
        for i in range(1, 21):
            db.execute("insert into t(id, v) values (?, ?)", [f"row{i}", _vec(i)])
            db.execute("insert into ref(id, v) values (?, ?)", [f"row{i}", _vec(i)])
    else:
        db.execute(f"create virtual table t using vec0(v float[8]{opts})")
        db.execute("create virtual table ref using vec0(v float[8])")
        for i in range(1, 21):
            db.execute("insert into t(rowid, v) values (?, ?)", [i, _vec(i)])
            db.execute("insert into ref(rowid, v) values (?, ?)", [i, _vec(i)])
    return db.execute("select v from ref where rowid = 1").fetchone()[0] \
        if not text_pk else db.execute("select v from ref where id = 'row1'").fetchone()[0]


@pytest.mark.parametrize("index", ["flat", "hnsw", "diskann", "rescore", "ivf"])
def test_filtered_knn_small_filter_is_exact(db, index):
    """A small filter ranks the rows it names exactly, as flat does."""
    q = _build(db, index)
    for rowids in ["9, 3, 5, 12", "2, 19, 20", "3, 3, 999"]:
        sql = f"select rowid from {{}} where v match ? and k = 3 and rowid in ({rowids})"
        got = [r[0] for r in db.execute(sql.format("t"), [q])]
        want = [r[0] for r in db.execute(sql.format("ref"), [q])]
        assert got == want


@pytest.mark.parametrize("index", ["flat", "hnsw", "diskann", "rescore", "ivf"])
def test_filtered_knn_text_primary_key(db, index):
    """Text-pk resolution (vec0_rowid_from_id) goes through the shared rowid
    collection for every index: `id in (...)` names text ids, and an id the
    table does not have is skipped."""
    q = _build(db, index, text_pk=True)
    sql = "select id from {} where v match ? and k = 3 and id in ('row9', 'row3', 'row5', 'missing')"
    got = [r[0] for r in db.execute(sql.format("t"), [q])]
    want = [r[0] for r in db.execute(sql.format("ref"), [q])]
    assert got == want == ["row3", "row5", "row9"]


@pytest.mark.parametrize("index", ["flat", "hnsw", "diskann", "rescore", "ivf"])
def test_filtered_knn_fewer_than_k_matches(db, index):
    """Caveat, stated so it cannot regress silently: when fewer than k rows
    match the filter, the query returns only those rows. It never pads the
    result with rows outside the filter."""
    q = _build(db, index)
    rows = db.execute(
        "select rowid from t where v match ? and k = 5 and rowid in (4, 7)", [q]
    ).fetchall()
    assert sorted(r[0] for r in rows) == [4, 7]
    rows = db.execute(
        "select rowid from t where v match ? and k = 5 and rowid in (999)", [q]
    ).fetchall()
    assert rows == []


@pytest.mark.parametrize("index", ["hnsw", "diskann", "ivf"])
def test_filtered_knn_large_filter_searches_with_filter(db, index):
    """A filter above VEC0_KNN_EXACT_FILTER_MAX is applied during traversal
    (here via a subquery, the shape the graph path uses), not ranked exactly.

    Caveat: a selective large filter can legitimately return fewer than k
    rows, because the traversal may not visit k allowed rows. Callers must
    not assume len(result) == k."""
    if index == "ivf" and not _has_ivf(db):
        pytest.skip("IVF not enabled (compile with -DSQLITE_VEC_EXPERIMENTAL_IVF_ENABLE=1)")
    db.execute(f"create virtual table t using vec0(v float[8]{INDEX_OPTS[index]})")
    rng = random.Random(7)
    db.executemany(
        "insert into t(rowid, v) values (?, ?)",
        [(i, _f32([rng.random() * 10 for _ in range(8)])) for i in range(5000)],
    )
    db.execute("create temp table keep(id integer primary key)")
    # 4500 allowed rows exceeds VEC0_KNN_EXACT_FILTER_MAX (4096), so the
    # index is searched with the filter applied instead of ranking exactly.
    db.executemany(
        "insert into keep values (?)", [(i,) for i in range(5000) if i % 10 != 0]
    )
    rows = db.execute(
        "select rowid, distance from t where v match ? and k = 5 "
        "and rowid in (select id from keep)",
        [_vec(1)],
    ).fetchall()
    assert 1 <= len(rows) <= 5
    assert all(r[0] % 10 != 0 for r in rows)
    assert [r[1] for r in rows] == sorted(r[1] for r in rows)
