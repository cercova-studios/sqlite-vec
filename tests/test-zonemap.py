"""Conservative chunk pruning must preserve the ordinary SQL filter result."""
import sqlite3
import struct
import random

import pytest

from helpers import _f32


def build(db):
    db.execute("create virtual table t using vec0(v float[1], n integer, f float, b boolean, chunk_size=8)")
    db.execute("create table truth(id integer primary key, n integer, f real, b integer)")
    for i in range(24):
        n = (i // 8) * 100 + i % 8
        db.execute("insert into t(rowid,v,n,f,b) values (?,?,?,?,?)",
                   [i + 1, _f32([i + 1]), n, n + .5, (i // 8) % 2])
        db.execute("insert into truth values (?,?,?,?)", [i + 1, n, n + .5, (i // 8) % 2])


def query(db, predicate, args=()):
    return [r[0] for r in db.execute(
        f"select rowid from t where v match '[0]' and k=24 and {predicate}", args)]


@pytest.mark.parametrize("column,targets,ops", [
    ("n", [-1, 0, 7, 8, 100, 107, 208, 2**63 - 1], ["=", "!=", "<", "<=", ">", ">="]),
    ("f", [-float("inf"), .5, 7.5, 100.5, 107.5, float("inf")], ["=", "!=", "<", "<=", ">", ">="]),
    ("b", [0, 1], ["=", "!="]),
])
def test_zonemap_comparisons_match_sql(db, column, targets, ops):
    build(db)
    for op in ops:
        for target in targets:
            predicate = f"{column} {op} ?"
            expected = [r[0] for r in db.execute(f"select id from truth where {predicate} order by id", [target])]
            assert query(db, predicate, [target]) == expected, (column, op, target)
    assert query(db, "n in (-100, 5, 102, 206, 999)") == [6, 11, 23]
    assert query(db, "n in (50, 150)") == []
    assert query(db, "n >= 100 and n < 108 and f > 100.5 and b = 1") == list(range(10, 17))


def test_zonemap_skips_excluded_metadata_blob(db):
    build(db)
    chunk = db.execute("select chunk_id from t_rowids where rowid=9").fetchone()[0]
    db.execute("update t_metadatachunks00 set data=x'00' where rowid=?", [chunk])
    assert query(db, "n < 8") == list(range(1, 9))
    db.execute("delete from t_info where key=?", [f"zonemap_00_{chunk}"])
    with pytest.raises(sqlite3.Error, match="filter metadata"):
        query(db, "n < 8")


def test_zonemap_widens_and_rolls_back(db):
    build(db)
    db.commit()
    original = [tuple(r) for r in db.execute("select key,value from t_info where key like 'zonemap_%' order by key")]
    assert len(original) == 9
    db.execute("begin")
    db.execute("update t set n=-50, f=-50.5 where rowid=1")
    assert query(db, "n = -50") == [1]
    db.execute("savepoint s")
    db.execute("update t set n=999 where rowid=2")
    assert query(db, "n = 999") == [2]
    db.execute("rollback to s")
    db.execute("release s")
    assert query(db, "n = 999") == []
    db.execute("rollback")
    assert [tuple(r) for r in db.execute("select key,value from t_info where key like 'zonemap_%' order by key")] == original
    db.execute("update t set n=-50 where rowid=1")
    chunk = db.execute("select chunk_id from t_rowids where rowid=1").fetchone()[0]
    bounds = db.execute("select value from t_info where key=?", [f"zonemap_00_{chunk}"]).fetchone()[0]
    assert struct.unpack("qq", bounds) == (-50, 7)
    db.execute("delete from t where rowid=1")
    assert query(db, "n = -50") == []
    assert db.execute("select value from t_info where key=?", [f"zonemap_00_{chunk}"]).fetchone()[0] == bounds
    db.execute("delete from t where rowid <= 8")
    assert db.execute("select value from t_info where key=?", [f"zonemap_00_{chunk}"]).fetchone() is None


def test_zonemap_missing_or_malformed_stats_stay_safe(db):
    build(db)
    db.execute("delete from t_info where key like 'zonemap_%'")
    db.execute("update t set n=-1000 where rowid=1")
    assert query(db, "n = -1000") == [1]
    assert db.execute("select count(*) from t_info where key like 'zonemap_%'").fetchone()[0] == 0
    db.execute("insert into t(rowid,v,n,f,b) values (25,'[25]',9999,9999.5,1)")
    assert query(db, "n >= 0") == list(range(2, 26))[:24]
    chunk = db.execute("select chunk_id from t_rowids where rowid=25").fetchone()[0]
    db.execute("update t_info set value=x'00' where key=?", [f"zonemap_00_{chunk}"])
    assert query(db, "n = 9999") == [25]
    db.execute("update t_info set value=? where key=?",
               [struct.pack("qq", 10000, -10000), f"zonemap_00_{chunk}"])
    assert query(db, "n = 9999") == [25]


def test_zonemap_i64_extremes_and_failed_statement(db):
    build(db)
    db.commit()
    db.execute("update t set n=? where rowid in (1,2)", [2**63 - 2])
    db.execute("update t set n=? where rowid=2", [2**63 - 1])
    assert query(db, "n != ?", [2**63 - 2]) == list(range(2, 25))
    assert query(db, "n = ?", [2**63 - 1]) == [2]
    assert query(db, "n in (?,?)", [2**63 - 2, 2**63 - 1]) == [1, 2]
    db.execute("create table input(n)")
    db.executemany("insert into input values (?)", [(-2**63,), ("wrong type",)])
    db.commit()
    before = [tuple(r) for r in db.execute("select key,value from t_info where key like 'zonemap_%' order by key")]
    with pytest.raises(sqlite3.Error, match="Expected integer"):
        db.execute("insert into t(v,n,f,b) select '[50]',n,0.5,1 from input order by rowid")
    assert [tuple(r) for r in db.execute("select key,value from t_info where key like 'zonemap_%' order by key")] == before
    assert query(db, "n = ?", [-2**63]) == []


def test_zonemap_chunk_reuse_and_two_connections(tmp_path):
    def connect():
        db = sqlite3.connect(tmp_path / "app.db")
        db.enable_load_extension(True)
        db.load_extension("dist/vec0")
        db.enable_load_extension(False)
        return db

    a = connect()
    build(a)
    a.commit()
    b = connect()
    assert query(b, "n > 1000") == []
    a.execute("update t set n=5000 where rowid=9")
    a.commit()
    assert query(b, "n = 5000") == [9]
    a.execute("delete from t")
    a.commit()
    assert a.execute("select count(*) from t_info where key like 'zonemap_%'").fetchone()[0] == 0
    a.execute("insert into t(rowid,v,n,f,b) values (1,'[1]',-100,-100.5,0)")
    a.commit()
    assert query(b, "n = -100") == [1]
    b.close()
    # Prepared stats statements must not retain the old table name on rename.
    a.execute("alter table t rename to renamed")
    a.execute("update renamed set n=-200 where rowid=1")
    a.commit()
    a.close()
    a = connect()
    assert a.execute("select rowid from renamed where v match '[0]' and k=1 and n=-200").fetchone() == (1,)
    a.close()


def test_zonemap_randomized_writes_match_sql(db):
    build(db)
    db.commit()
    rng = random.Random(414)
    next_id = 25
    for step in range(100):
        live = [r[0] for r in db.execute("select id from truth")]
        action = rng.choice(["insert", "update", "delete"])
        n = rng.choice([-2**63, 2**63 - 1, rng.randrange(-1000, 1000)])
        f, b = rng.uniform(-1000, 1000), rng.randrange(2)
        if action == "insert" or not live:
            db.execute("insert into t(rowid,v,n,f,b) values (?,?,?,?,?)", [next_id, _f32([next_id]), n, f, b])
            db.execute("insert into truth values (?,?,?,?)", [next_id, n, f, b])
            next_id += 1
        else:
            rowid = rng.choice(live)
            if action == "update":
                db.execute("update t set n=?, f=?, b=? where rowid=?", [n, f, b, rowid])
                db.execute("update truth set n=?, f=?, b=? where id=?", [n, f, b, rowid])
            else:
                db.execute("delete from t where rowid=?", [rowid])
                db.execute("delete from truth where id=?", [rowid])
        for column, target, ops in [("n", n, ["=", "!=", "<", "<=", ">", ">="]),
                                     ("f", f, ["=", "!=", "<", "<=", ">", ">="]),
                                     ("b", b, ["=", "!="])]:
            for op in ops:
                predicate = f"{column} {op} ?"
                expected = [r[0] for r in db.execute(
                    f"select id from truth where {predicate} order by id limit 24", [target])]
                assert query(db, predicate, [target]) == expected, (step, column, op, target)
        if step % 10 == 0:
            db.rollback()
        elif step % 10 == 5:
            db.commit()
