import random
import sqlite3
import struct

import pytest


def _f32(v):
    return struct.pack(f"{len(v)}f", *v)


def connect(path=":memory:"):
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    db.load_extension("dist/vec0")
    db.enable_load_extension(False)
    return db


@pytest.fixture()
def db():
    return connect()


def clustered(rng, n, d, centers=20, spread=0.3):
    cs = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(centers)]
    return [[x + rng.gauss(0, spread) for x in rng.choice(cs)] for _ in range(n)]


def recall(db, table, ref, queries, k, conv="?"):
    hits = 0
    for q in queries:
        sql = f"select rowid from {{}} where v match {conv} and k = ?"
        got = {r[0] for r in db.execute(sql.format(table), [q, k])}
        want = {r[0] for r in db.execute(sql.format(ref), [q, k])}
        hits += len(got & want)
    return hits / (len(queries) * k)


def build(db, d, rows, metric="l2", elem="float", opts="", conv="?"):
    dm = "" if elem == "bit" else f"distance_metric={metric}"
    for name, idx in [("t", f"indexed by hnsw({opts})"), ("ref", "")]:
        db.execute(f"create virtual table {name} using vec0(v {elem}[{d}] {dm} {idx})")
        db.executemany(f"insert into {name}(rowid, v) values (?, {conv})", list(enumerate(rows)))


def unit(v):
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


@pytest.mark.parametrize("metric", ["l2", "cosine", "dot", "l1"])
def test_hnsw_recall_float(db, metric):
    rng = random.Random(0)
    # dot is only a sound graph metric on unit vectors (same ranking as cosine)
    prep = unit if metric == "dot" else (lambda v: v)
    rows = [_f32(prep(v)) for v in clustered(rng, 1500, 32)]
    build(db, 32, rows, metric=metric)
    queries = [_f32(prep(v)) for v in clustered(rng, 30, 32)]
    assert recall(db, "t", "ref", queries, 10) >= 0.95


def test_hnsw_recall_int8_and_bit(db):
    rng = random.Random(1)
    ints = [str([rng.randint(-100, 100) for _ in range(16)]) for _ in range(800)]
    build(db, 16, ints, elem="int8", conv="vec_int8(?)")
    qs = [str([rng.randint(-100, 100) for _ in range(16)]) for _ in range(20)]
    assert recall(db, "t", "ref", qs, 5, conv="vec_int8(?)") >= 0.9

    d2 = connect()
    bits = [bytes(rng.getrandbits(8) for _ in range(8)) for _ in range(800)]
    build(d2, 64, bits, elem="bit", conv="vec_bit(?)")
    qs = [bytes(rng.getrandbits(8) for _ in range(8)) for _ in range(20)]
    # hamming distances tie a lot; compare distances, not rowids
    for q in qs:
        sql = "select distance from {} where v match vec_bit(?) and k = 5"
        got = [r[0] for r in d2.execute(sql.format("t"), [q])]
        want = [r[0] for r in d2.execute(sql.format("ref"), [q])]
        assert got[-1] <= want[-1] + 2


def test_hnsw_results_and_point_reads(db):
    db.execute("create virtual table t using vec0(v float[2] indexed by hnsw(m=4))")
    db.executemany("insert into t(rowid, v) values (?, ?)",
                   [(i, _f32([i, 0])) for i in range(1, 51)])
    rows = db.execute("select rowid, distance from t where v match ? and k = 3",
                      [_f32([10.2, 0])]).fetchall()
    assert [r[0] for r in rows] == [10, 11, 9]
    assert rows[0][1] == pytest.approx(0.2, rel=1e-4)
    assert db.execute("select v from t where rowid = 7").fetchone()[0] == _f32([7, 0])
    # k larger than the table
    assert len(db.execute("select rowid from t where v match ? and k = 500",
                          [_f32([0, 0])]).fetchall()) == 50
    # empty graph
    db.execute("create virtual table e using vec0(v float[2] indexed by hnsw())")
    assert db.execute("select * from e where v match ? and k = 3", [_f32([0, 0])]).fetchall() == []


def test_hnsw_delete_keeps_recall(db):
    rng = random.Random(2)
    rows = [_f32(v) for v in clustered(rng, 1200, 24)]
    build(db, 24, rows)
    entry = db.execute("select rowid from t_hnsw_nodes00 order by level desc, rowid limit 1").fetchone()[0]
    gone = set(rng.sample(range(1200), 400)) | {entry}
    for name in ("t", "ref"):
        db.executemany(f"delete from {name} where rowid = ?", [(r,) for r in gone])
    assert db.execute("select count(*) from t_hnsw_nodes00").fetchone()[0] == 1200 - len(gone)
    # no edges of deleted nodes remain as rows
    assert db.execute("select count(*) from t_hnsw_edges00 where rowid not in "
                      "(select rowid from t_hnsw_nodes00)").fetchone()[0] == 0
    queries = [_f32(v) for v in clustered(rng, 30, 24)]
    assert recall(db, "t", "ref", queries, 10) >= 0.9
    for q in queries[:5]:
        got = {r[0] for r in db.execute("select rowid from t where v match ? and k = 10", [q])}
        assert not (got & gone)


def test_hnsw_update(db):
    db.execute("create virtual table t using vec0(v float[2] indexed by hnsw(m=4))")
    db.executemany("insert into t(rowid, v) values (?, ?)", [(i, _f32([i, i])) for i in range(30)])
    db.execute("update t set v = ? where rowid = 3", [_f32([100, 100])])
    assert db.execute("select rowid from t where v match ? and k = 1", [_f32([99, 99])]).fetchone()[0] == 3
    assert db.execute("select v from t where rowid = 3").fetchone()[0] == _f32([100, 100])
    with pytest.raises(sqlite3.OperationalError, match="must be"):
        db.execute("update t set v = ? where rowid = 3", [_f32([1, 2, 3])])


def test_hnsw_rowid_filters_and_distance(db):
    rng = random.Random(3)
    rows = [_f32(v) for v in clustered(rng, 6000, 8, centers=5)]
    build(db, 8, rows, opts="m=8, ef_construction=64")
    q = rows[0]
    # small filter: ranked exactly
    allowed = sorted(rng.sample(range(6000), 50))
    marks = ",".join("?" * len(allowed))
    sql = f"select rowid from {{}} where v match ? and k = 5 and rowid in ({marks})"
    assert [r[0] for r in db.execute(sql.format("t"), [q, *allowed])] == \
           [r[0] for r in db.execute(sql.format("ref"), [q, *allowed])]
    # large filter: searched with the filter applied during traversal
    allowed = list(range(0, 6000, 2))
    db.execute("create temp table keep(id integer primary key)")
    db.executemany("insert into keep values (?)", [(a,) for a in allowed])
    got = [r[0] for r in db.execute(
        "select rowid from t where v match ? and k = 10 and rowid in (select id from keep)", [q])]
    assert len(got) == 10 and all(g % 2 == 0 for g in got)
    # distance constraint
    far = db.execute("select distance from ref where v match ? and k = 20", [q]).fetchall()[-1][0]
    got = db.execute("select distance from t where v match ? and k = 50 and distance < ?", [q, far]).fetchall()
    assert got and all(d < far for (d,) in got)


def test_hnsw_ef_search_command(db):
    rng = random.Random(4)
    rows = [_f32(v) for v in clustered(rng, 1500, 32, spread=1.0)]
    build(db, 32, rows, opts="m=4, ef_construction=16, ef_search=1")
    queries = [_f32(v) for v in clustered(rng, 30, 32, spread=1.0)]
    low = recall(db, "t", "ref", queries, 10)
    db.execute("insert into t(t) values ('ef_search=200')")
    assert recall(db, "t", "ref", queries, 10) >= max(low, 0.9)
    with pytest.raises(sqlite3.OperationalError):
        db.execute("insert into t(t) values ('ef_search=0')")


def test_hnsw_transactions_rename_drop(tmp_path):
    path = str(tmp_path / "h.db")
    db = connect(path)
    db.execute("create virtual table t using vec0(v float[2] indexed by hnsw())")
    db.executemany("insert into t(rowid, v) values (?, ?)", [(i, _f32([i, 0])) for i in range(20)])
    db.commit()
    db.execute("insert into t(rowid, v) values (99, ?)", [_f32([5.1, 0])])
    db.rollback()
    q = _f32([5.1, 0])
    assert db.execute("select rowid from t where v match ? and k = 1", [q]).fetchone()[0] == 5
    db.close()

    db = connect(path)  # graph persisted with the database, no load step
    assert db.execute("select rowid from t where v match ? and k = 1", [q]).fetchone()[0] == 5
    db.execute("alter table t rename to t2")
    names = {r[0] for r in db.execute("select name from sqlite_master")}
    assert {"t2_hnsw_nodes00", "t2_hnsw_edges00"} <= names
    assert db.execute("select rowid from t2 where v match ? and k = 1", [q]).fetchone()[0] == 5
    db.execute("drop table t2")
    assert not [n for (n,) in db.execute("select name from sqlite_master") if "hnsw" in n]


def test_hnsw_bad_options(db):
    for opts in ["m=1", "m=999", "ef_search=0", "nope=3", "m=abc"]:
        with pytest.raises(sqlite3.OperationalError):
            db.execute(f"create virtual table x using vec0(v float[2] indexed by hnsw({opts}))")
    with pytest.raises(sqlite3.OperationalError, match="HNSW"):
        db.execute("create virtual table x using vec0(v float[2] indexed by hnsw(), m text)")


def _nearest(db, q, table="t"):
    return db.execute(f"select rowid, distance from {table} where v match ? and k = 1", [q]).fetchone()


def test_hnsw_cache_sees_other_connections(tmp_path):
    path = str(tmp_path / "c.db")
    a, b = connect(path), connect(path)
    a.execute("create virtual table t using vec0(v float[2] indexed by hnsw(m=4))")
    a.executemany("insert into t(rowid, v) values (?, ?)", [(i, _f32([i, 0])) for i in range(40)])
    a.commit()
    q = _f32([10.1, 0])
    assert _nearest(a, q)[0] == 10  # warms a's cache
    b.execute("delete from t where rowid = 10")
    b.execute("update t set v = ? where rowid = 3", [_f32([10.2, 0])])
    b.commit()
    assert _nearest(a, q)[0] == 3
    assert a.execute("select v from t where rowid = 3").fetchone()[0] == _f32([10.2, 0])


def test_hnsw_cache_statement_rollback(db):
    db.isolation_level = None  # explicit transactions only
    db.execute("create virtual table t using vec0(v float[2] indexed by hnsw(m=4))")
    db.executemany("insert into t(rowid, v) values (?, ?)", [(i, _f32([i, 0])) for i in range(20)])
    db.execute("create table src(id, v)")
    db.executemany("insert into src values (?, ?)", [(100, _f32([5.1, 0])), (101, _f32([5.2, 0])), (1, _f32([0, 0]))])
    db.execute("begin")
    _nearest(db, _f32([5.1, 0]))  # warm
    with pytest.raises(sqlite3.Error):
        # rows 100 and 101 go in, then rowid 1 collides: the statement rolls back
        db.execute("insert into t(rowid, v) select id, v from src order by id desc")
    assert _nearest(db, _f32([5.1, 0]))[0] == 5
    db.execute("insert into t(rowid, v) values (100, ?)", [_f32([50, 0])])
    assert _nearest(db, _f32([5.1, 0]))[0] == 5
    assert _nearest(db, _f32([49, 0]))[0] == 100
    db.execute("savepoint s")
    db.execute("delete from t where rowid = 5")
    db.execute("rollback to s")
    assert _nearest(db, _f32([5.1, 0]))[0] == 5
    db.execute("commit")
    assert db.execute("select count(*) from t_hnsw_nodes00").fetchone()[0] == 21


def test_hnsw_cache_disabled(db):
    db.execute("create virtual table t using vec0(v float[2] indexed by hnsw(cache_mb=0))")
    db.executemany("insert into t(rowid, v) values (?, ?)", [(i, _f32([i, 0])) for i in range(30)])
    assert _nearest(db, _f32([7.2, 0]))[0] == 7
    db.execute("insert into t(t) values ('cache_mb=8')")
    assert _nearest(db, _f32([7.2, 0]))[0] == 7
    with pytest.raises(sqlite3.OperationalError):
        db.execute("insert into t(t) values ('cache_mb=-1')")


@pytest.mark.parametrize("metric", ["l2", "cosine", "dot"])
def test_hnsw_int8_search_codes(db, metric):
    """quantizer=int8 searches on codes, then reranks with exact distances."""
    rng = random.Random(5)
    prep = unit if metric == "dot" else (lambda v: v)
    rows = [_f32(prep(v)) for v in clustered(rng, 1500, 48)]
    build(db, 48, rows, metric=metric, opts="quantizer=int8")
    queries = [_f32(prep(v)) for v in clustered(rng, 30, 48)]
    assert recall(db, "t", "ref", queries, 10) >= 0.95
    # reported distances are exact, not approximations
    q = queries[0]
    got = db.execute("select rowid, distance from t where v match ? and k = 3", [q]).fetchall()
    exact = dict(db.execute("select rowid, distance from ref where v match ? and k = 50", [q]).fetchall())
    for rowid, d in got:
        assert d == pytest.approx(exact[rowid], rel=1e-6)
    # writes keep codes in step with vectors
    db.execute("delete from t where rowid = ?", [got[0][0]])
    db.execute("update t set v = ? where rowid = ?", [q, got[1][0]])
    assert db.execute("select rowid from t where v match ? and k = 1", [q]).fetchone()[0] == got[1][0]


def test_hnsw_quantizer_option(db):
    for bad in ["quantizer=int4", "quantizer=3", "m=int8"]:
        with pytest.raises(sqlite3.OperationalError):
            db.execute(f"create virtual table x using vec0(v float[8] indexed by hnsw({bad}))")
    # ignored where codes do not apply: int8/bit columns, l1
    for col in ["v int8[8]", "v bit[8]", "v float[8] distance_metric=l1"]:
        db.execute(f"create virtual table y using vec0({col} indexed by hnsw(quantizer=int8))")
        db.execute("drop table y")


def test_hnsw_rebuild(db):
    rng = random.Random(6)
    rows = [_f32(v) for v in clustered(rng, 1200, 24)]
    build(db, 24, rows, opts="quantizer=int8")
    gone = rng.sample(range(1200), 400)
    for name in ("t", "ref"):
        db.executemany(f"delete from {name} where rowid = ?", [(r,) for r in gone])
    db.execute("insert into t(t) values ('hnsw-rebuild')")
    # every list names only live nodes, and every live node has a level-0 list
    live = {r[0] for r in db.execute("select rowid from t_hnsw_nodes00")}
    for (blob,) in db.execute("select neighbors from t_hnsw_edges00"):
        assert set(struct.unpack(f"{len(blob)//8}q", blob)) <= live
    assert db.execute("select count(*) from t_hnsw_edges00 where level = 0").fetchone()[0] == len(live)
    queries = [_f32(v) for v in clustered(rng, 30, 24)]
    assert recall(db, "t", "ref", queries, 10) >= 0.95

    # a rolled-back rebuild leaves the old graph
    before = db.execute("select count(*), sum(length(neighbors)) from t_hnsw_edges00").fetchone()
    db.commit()
    db.execute("delete from t_hnsw_edges00 where rowid in (select rowid from t_hnsw_nodes00 limit 50)")
    db.rollback()
    db.execute("begin")
    db.execute("insert into t(t) values ('hnsw-rebuild')")
    db.rollback()
    assert db.execute("select count(*), sum(length(neighbors)) from t_hnsw_edges00").fetchone() == before
    assert recall(db, "t", "ref", queries, 10) >= 0.95

    db.execute("create virtual table e using vec0(v float[2] indexed by hnsw())")
    db.execute("insert into e(e) values ('hnsw-rebuild')")  # empty graph
