import sqlite3
import struct
import random

import pytest


def connect(path=":memory:"):
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    db.load_extension("dist/vec0")
    return db


def create(db, quantizer="none", name="t"):
    oversample = "" if quantizer == "none" else ",oversample=10"
    db.execute(f"create virtual table {name} using vec0(v float[8] "
               f"indexed by ivf(nlist=2,nprobe=2,quantizer={quantizer}{oversample}))")


def vector(i):
    return struct.pack("8f", i, *([0] * 7))


def test_ivf_defers_cell_writes():
    db = connect()
    create(db)
    db.commit()
    db.executemany("insert into t(rowid,v) values (?,?)", [(i, vector(i)) for i in range(1, 41)])
    assert db.execute("select sum(n_vectors) from t_ivf_cells00").fetchone() == (0,)
    assert db.execute("select rowid from t where v match ? and k=3", [vector(20.1)]).fetchall() == [(20,), (21,), (19,)]
    assert db.execute("select sum(n_vectors) from t_ivf_cells00").fetchone() == (40,)
    db.commit()
    db.close()


@pytest.mark.parametrize("quantizer", ["none", "int8", "binary", "int4", "rabitq", "fp16", "bf16"])
def test_ivf_reuses_deleted_slot_without_overwriting_live_row(quantizer):
    db = connect()
    create(db, quantizer)
    db.executemany("insert into t(rowid,v) values (?,?)", [(i, vector(i)) for i in range(1, 11)])
    db.commit()
    db.execute("delete from t where rowid=3")
    db.execute("insert into t(rowid,v) values (20,?)", [vector(20)])
    for i in [1, 2, *range(4, 11), 20]:
        assert db.execute("select v from t where rowid=?", [i]).fetchone() == (vector(i),)
    got = db.execute("select rowid from t where v match ? and k=10", [vector(10)]).fetchall()
    assert {r[0] for r in got} == {1, 2, *range(4, 11), 20}
    db.commit()
    db.close()


@pytest.mark.parametrize("when", ["read", "commit", "savepoint"])
def test_ivf_flush_failure(when):
    db = connect()
    create(db)
    db.execute("create trigger fault before update on t_ivf_cells00 when new.cell_id=2 "
               "begin select raise(abort,'cell fault'); end")
    db.commit()
    db.executemany("insert into t(rowid,v) values (?,?)", [(i, vector(i)) for i in range(1, 131)])
    with pytest.raises(sqlite3.DatabaseError) as error:
        if when == "read":
            db.execute("select v from t").fetchall()
        elif when == "savepoint":
            db.execute("savepoint s")
        else:
            db.commit()
    if when == "savepoint":
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT
    else:
        assert "cell fault" in str(error.value)
    if when != "commit":
        assert db.in_transaction
        db.execute("drop trigger fault")
        # Retrying after the nested shadow UPDATE fails must retain all vectors.
        got = dict(db.execute("select rowid,v from t"))
        assert got == {i: vector(i) for i in range(1, 131)}
        db.commit()
    else:
        assert not db.in_transaction
        assert db.execute("select count(*) from t").fetchone() == (0,)
    db.close()


@pytest.mark.parametrize("trained", [False, True])
@pytest.mark.parametrize("quantizer", ["none", "int8", "binary", "int4", "rabitq", "fp16", "bf16"])
def test_ivf_batch_savepoints_and_commands(tmp_path, quantizer, trained):
    path = tmp_path / "batch.db"
    db = connect(path)
    create(db, quantizer)
    db.executemany("insert into t(rowid,v) values (?,?)", [(i, vector(i)) for i in range(1, 21)])
    if trained:
        db.execute("insert into t(t) values ('compute-centroids')")
    db.commit()
    before = db.execute("select * from t_ivf_cells00 order by cell_id").fetchall()
    db.execute("insert into t(rowid,v) values (30,?)", [vector(30)])
    db.execute("savepoint outer_s")
    db.execute("delete from t where rowid=4")
    db.execute("insert into t(rowid,v) values (40,?)", [vector(40)])
    db.execute("savepoint inner_s")
    db.execute("delete from t where rowid=30")
    db.execute("release inner_s")
    db.execute("rollback to outer_s")
    assert dict(db.execute("select rowid,v from t")) == {i: vector(i) for i in [*range(1, 21), 30]}
    db.execute("release outer_s")
    db.execute("insert into t(rowid,v) values (50,?)", [vector(50)])
    with pytest.raises(sqlite3.Error):
        db.execute("insert into t(rowid,v) values (60,?), (1,?)", [vector(60), vector(1)])
    assert db.execute("select v from t where rowid=50").fetchone() == (vector(50),)
    assert db.execute("select v from t where rowid=60").fetchone() is None
    got = db.execute("select rowid from t where v match ? and k=3 and rowid in (1,20,30)", [vector(30)]).fetchall()
    assert got == [(30,), (20,), (1,)]
    db.rollback()
    assert db.execute("select * from t_ivf_cells00 order by cell_id").fetchall() == before
    db.execute("insert into t(rowid,v) values (30,?)", [vector(30)])
    db.execute("alter table t rename to renamed")
    db.commit()
    db.close()
    db = connect(path)
    assert db.execute("select v from renamed where rowid=30").fetchone() == (vector(30),)
    db.close()


@pytest.mark.parametrize("dimensions,rows", [(8, 4200), (4096, 600)])
def test_ivf_batch_pressure_and_rollback(dimensions, rows):
    db = connect()
    db.execute(f"create virtual table t using vec0(v float[{dimensions}] indexed by ivf())")
    db.commit()
    v = struct.pack(f"{dimensions}f", *([1.] * dimensions))
    db.executemany("insert into t(rowid,v) values (?,?)", ((i, v) for i in range(rows)))
    # Both the cell-count and byte cap must flush before the final commit.
    assert db.execute("select sum(n_vectors) from t_ivf_cells00").fetchone()[0] > 0
    assert db.execute("select count(*) from t where v match ? and k=4096", [v]).fetchone() == (min(rows, 4096),)
    assert db.execute("select v from t where rowid=?", [rows-1]).fetchone() == (v,)
    db.execute("savepoint s")
    db.executemany("insert into t(rowid,v) values (?,?)", ((i + rows, v) for i in range(rows)))
    db.execute("rollback to s")
    db.execute("release s")
    assert db.execute("select count(*) from t").fetchone() == (rows,)
    db.rollback()
    assert db.execute("select count(*) from t").fetchone() == (0,)
    db.close()


def test_ivf_batch_release_keeps_pending_and_rollback(tmp_path):
    db = connect(tmp_path / "release.db")
    create(db)
    db.execute("create trigger fault before update on t_ivf_cells00 "
               "begin select raise(abort,'cell fault'); end")
    db.commit()
    db.execute("begin")
    db.execute("savepoint outer_s")
    db.execute("savepoint inner_s")
    db.executemany("insert into t(rowid,v) values (?,?)", [(i, vector(i)) for i in range(1, 20)])
    db.execute("release inner_s")
    db.execute("rollback to outer_s")
    db.execute("release outer_s")
    db.commit()
    assert db.execute("select count(*) from t").fetchone() == (0,)
    db.close()


def test_ivf_batch_other_connection_and_multicolumn(tmp_path):
    path = tmp_path / "connections.db"
    a, b = connect(path), connect(path)
    a.execute("create virtual table t using vec0("
              "v float[8] indexed by ivf(), w float[8] indexed by ivf(quantizer=int8))")
    a.commit()
    a.executemany("insert into t(rowid,v,w) values (?,?,?)", [(i, vector(i), vector(-i)) for i in range(1, 21)])
    assert b.execute("select count(*) from t").fetchone() == (0,)
    a.commit()
    assert b.execute("select v,w from t where rowid=20").fetchone() == (vector(20), vector(-20))
    b.execute("delete from t where rowid=4")
    b.commit()
    assert a.execute("select v from t where rowid=4").fetchone() is None
    a.close()
    b.close()


def test_ivf_flush_failure_preserves_pending_hnsw():
    db = connect()
    db.execute("create virtual table t using vec0("
               "v float[8] indexed by ivf(),w float[8] indexed by hnsw(m=4))")
    db.execute("create trigger fault before update on t_ivf_cells00 "
               "begin select raise(abort,'cell fault'); end")
    db.commit()
    db.executemany("insert into t(rowid,v,w) values (?,?,?)", [(i, vector(i), vector(i)) for i in range(1, 41)])
    with pytest.raises(sqlite3.DatabaseError, match="cell fault"):
        db.execute("select v from t").fetchall()
    db.execute("drop trigger fault")
    assert db.execute("select rowid from t where w match ? and k=1", [vector(20)]).fetchone() == (20,)
    db.commit()
    assert db.execute("select rowid from t where v match ? and k=1", [vector(20)]).fetchone() == (20,)
    db.close()


def test_ivf_batch_randomized_oracle():
    db = connect()
    create(db)
    rng = random.Random(2610)
    live = {}

    def verify():
        q = vector(.37)
        want = sorted(live, key=lambda i: abs(i - .37))[:5]
        assert [r[0] for r in db.execute("select rowid from t where v match ? and k=5", [q])] == want
        assert dict(db.execute("select rowid,v from t")) == live

    for turn in range(80):
        db.execute("savepoint s")
        old = dict(live)
        for _ in range(5):
            i = rng.randrange(1, 100)
            if i in live:
                db.execute("delete from t where rowid=?", [i])
                del live[i]
            else:
                db.execute("insert into t(rowid,v) values (?,?)", [i, vector(i)])
                live[i] = vector(i)
        verify()
        if turn % 3 == 0:
            db.execute("rollback to s")
            live = old
            verify()
        db.execute("release s")
    db.close()


def test_ivf_read_flush_interrupted_at_each_step():
    for stop in range(1, 1000):
        db = connect()
        create(db)
        db.execute("insert into t(rowid,v) values (1,?)", [vector(1)])
        db.commit()
        db.executemany("insert into t(rowid,v) values (?,?)", [(i, vector(i)) for i in range(2, 20)])
        steps = 0

        def interrupt():
            nonlocal steps
            steps += 1
            return steps == stop

        db.set_progress_handler(interrupt, 1)
        succeeded = False
        try:
            db.execute("select rowid,v from t").fetchall()
            succeeded = True
        except sqlite3.OperationalError as error:
            # Existing column-read helpers also map nested interrupts to ERROR
            # (reproduced on the baseline); all live vector values must survive.
            assert str(error) in ("interrupted", "SQL logic error")
        db.set_progress_handler(None, 0)
        want = range(1, 20) if db.in_transaction else [1]
        assert dict(db.execute("select rowid,v from t")) == {i: vector(i) for i in want}
        db.close()
        if succeeded:
            break
    else:
        pytest.fail("read flush did not finish within 1000 SQLite steps")
