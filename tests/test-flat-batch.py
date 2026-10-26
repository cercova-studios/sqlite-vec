import sqlite3
import struct
import random

import pytest


def connect(path=":memory:"):
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    db.load_extension("dist/vec0")
    return db


def vector(x):
    return struct.pack("2f", x, 0)


def test_flat_vectors_deferred_until_read_or_commit(tmp_path):
    path = tmp_path / "batch.db"
    db = connect(path)
    db.execute("create virtual table t using vec0(v float[2],chunk_size=8)")
    db.commit()
    db.executemany("insert into t(rowid,v) values (?,?)", [(i, vector(i)) for i in range(1, 7)])
    assert db.execute("select vectors=zeroblob(64) from t_vector_chunks00").fetchone() == (1,)
    assert db.execute("select v from t where rowid=6").fetchone() == (vector(6),)
    db.execute("insert into t(rowid,v) values (7,?)", [vector(7)])
    db.commit()
    db.close()
    db = connect(path)
    assert db.execute("select v from t where rowid=7").fetchone() == (vector(7),)
    db.close()


@pytest.mark.parametrize("kind,conv", [("float[2]", "?"), ("int8[2]", "vec_int8(?)"), ("bit[8]", "vec_bit(?)")])
def test_flat_batch_types_savepoints_delete_reuse(tmp_path, kind, conv):
    def value(i):
        if kind.startswith("float"):
            return vector(i)
        if kind.startswith("int8"):
            return struct.pack("2b", i, -i)
        return bytes([i])
    path = tmp_path / "types.db"
    db = connect(path)
    db.execute(f"create virtual table t using vec0(v {kind},chunk_size=8)")
    db.commit()
    db.executemany(f"insert into t(rowid,v) values (?,{conv})", [(i, value(i)) for i in range(1, 20)])
    db.execute("savepoint s")
    if kind.startswith("float"):
        db.execute(f"update t set v={conv} where rowid=1", [value(50)])
    else:
        # Existing SQLite subtype loss rejects these UPDATEs on the baseline too.
        with pytest.raises(sqlite3.OperationalError, match="invalid float32 vector BLOB length"):
            db.execute(f"update t set v={conv} where rowid=1", [value(50)])
    db.execute("delete from t where rowid=2")
    db.execute(f"insert into t(rowid,v) values (30,{conv})", [value(30)])
    db.execute("savepoint inner_s")
    db.execute("delete from t where rowid=3")
    db.execute("release inner_s")
    db.execute("rollback to s")
    db.execute("release s")
    assert dict(db.execute("select rowid,v from t")) == {i: value(i) for i in range(1, 20)}
    db.execute(f"insert into t(rowid,v) values (30,{conv})", [value(30)])
    with pytest.raises(sqlite3.Error):
        db.execute(f"insert into t(rowid,v) values (31,{conv}),(1,{conv})", [value(31), value(1)])
    assert db.execute("select v from t where rowid=30").fetchone() == (value(30),)
    assert db.execute("select v from t where rowid=31").fetchone() is None
    db.execute("delete from t")
    db.execute(f"insert into t(rowid,v) values (1,{conv})", [value(50)])
    db.execute("alter table t rename to renamed")
    db.commit()
    db.close()
    db = connect(path)
    assert db.execute("select v from renamed where rowid=1").fetchone() == (value(50),)
    db.close()


@pytest.mark.parametrize("operation", ["read", "commit", "savepoint"])
def test_flat_batch_flush_failure(operation):
    db = connect()
    db.execute("create virtual table t using vec0(v float[2],chunk_size=8)")
    db.commit()
    db.executemany("insert into t(rowid,v) values (?,?)", [(i, vector(i)) for i in range(1, 7)])
    db.execute("pragma query_only=on")
    with pytest.raises(sqlite3.Error):
        if operation == "read":
            db.execute("select v from t").fetchall()
        elif operation == "commit":
            db.commit()
        else:
            db.execute("savepoint s")
    db.execute("pragma query_only=off")
    if operation == "commit":
        assert not db.in_transaction
        assert db.execute("select count(*) from t").fetchone() == (0,)
    else:
        assert dict(db.execute("select rowid,v from t")) == {i: vector(i) for i in range(1, 7)}
        db.commit()
    db.close()


def test_flat_batch_pressure_partitions_and_rollback():
    db = connect()
    db.execute("create virtual table t using vec0(v float[1024],p integer partition key,chunk_size=128)")
    db.commit()
    v = struct.pack("1024f", *([1.] * 1024))
    db.executemany("insert into t(rowid,v,p) values (?,?,?)", ((i, v, i % 2) for i in range(200)))
    # Alternating chunks and the byte cap force flushes before commit.
    assert any(blob != bytes(len(blob)) for (blob,) in db.execute("select vectors from t_vector_chunks00"))
    assert db.execute("select count(*) from t where v match ? and k=200 and p=1", [v]).fetchone() == (100,)
    db.execute("savepoint s")
    db.execute("delete from t where p=1")
    db.execute("rollback to s")
    db.execute("release s")
    assert db.execute("select count(*) from t").fetchone() == (200,)
    db.rollback()
    assert db.execute("select count(*) from t").fetchone() == (0,)
    db.close()


def test_flat_batch_randomized_oracle():
    db = connect()
    db.execute("create virtual table t using vec0(v float[2],score integer,chunk_size=8)")
    rng = random.Random(2606)
    live = {}
    for turn in range(80):
        db.execute("savepoint s")
        old = dict(live)
        for _ in range(5):
            rowid = rng.randrange(1, 100)
            if rowid in live:
                if rng.random() < .5:
                    db.execute("delete from t where rowid=?", [rowid])
                    del live[rowid]
                else:
                    x = rng.random() * 200
                    x = struct.unpack("f", struct.pack("f", x))[0]
                    db.execute("update t set v=? where rowid=?", [vector(x), rowid])
                    live[rowid] = x
            else:
                db.execute("insert into t(rowid,v,score) values (?,?,?)", [rowid, vector(rowid), rowid])
                live[rowid] = float(rowid)
        if turn % 3 == 0:
            db.execute("rollback to s")
            live = old
        for minimum in (0, 30):
            want = sorted((i for i in live if i >= minimum), key=lambda i: abs(live[i] - .37))[:5]
            got = db.execute("select rowid from t where v match ? and k=5 and score>=?", [vector(.37), minimum]).fetchall()
            assert [r[0] for r in got] == want
        db.execute("release s")
    db.close()


def test_flat_batch_capacity_and_visibility(tmp_path):
    path = tmp_path / "capacity.db"
    a, b = connect(path), connect(path)
    a.execute("create virtual table t using vec0(v float[1024],chunk_size=128)")
    a.commit()
    v = struct.pack("1024f", *([1.] * 1024))
    a.executemany("insert into t(rowid,v) values (?,?)", [(i, v) for i in range(40)])
    blob = a.execute("select vectors from t_vector_chunks00").fetchone()[0]
    # Each 64 KiB range holds 16 vectors: two ranges flushed, eight pending.
    assert blob[:32 * len(v)] == v * 32
    assert blob[32 * len(v):] == bytes(len(blob) - 32 * len(v))
    assert b.execute("select count(*) from t").fetchone() == (0,)
    a.commit()
    assert b.execute("select v from t where rowid=39").fetchone() == (v,)
    b.execute("update t set v=? where rowid=39", [bytes(len(v))])
    b.commit()
    assert a.execute("select v from t where rowid=39").fetchone() == (bytes(len(v)),)
    a.close()
    b.close()


def test_flat_batch_mixed_indexes():
    db = connect()
    ivf = "ivf" in db.execute("select vec_debug()").fetchone()[0].split("Build flags:")[-1]
    columns = ["v float[8]", "r float[8] indexed by rescore(quantizer=bit)", "h float[8] indexed by hnsw(m=4)"]
    if ivf:
        columns.append("i float[8] indexed by ivf()")
    db.execute(f"create virtual table t using vec0({','.join(columns)})")
    names = ["v", "r", "h"] + (["i"] if ivf else [])
    values = [struct.pack("8f", float(i), *([0.] * 7)) for i in range(1, 41)]
    placeholders = ','.join('?' for _ in names)
    db.executemany(f"insert into t(rowid,{','.join(names)}) values (?,{placeholders})",
                   [(i, *([v] * len(names))) for i, v in enumerate(values, start=1)])
    db.execute("savepoint s")
    db.execute("delete from t where rowid=20")
    db.execute("rollback to s")
    db.execute("release s")
    for name in names:
        assert db.execute(f"select {name} from t where rowid=20").fetchone() == (values[19],)
        # Include all tied sign codes before rescore's original-vector reranking.
        assert db.execute(f"select rowid from t where {name} match ? and k=40", [values[19]]).fetchone() == (20,)
    db.commit()
    db.close()


def test_flat_batch_read_interrupted_at_each_step():
    for stop in range(1, 1200):
        db = connect()
        db.execute("create virtual table t using vec0(v float[2],w float[2])")
        db.execute("insert into t(rowid,v,w) values (1,?,?)", [vector(1), vector(-1)])
        db.commit()
        db.executemany("insert into t(rowid,v,w) values (?,?,?)", [(i, vector(i), vector(-i)) for i in range(2, 20)])
        steps = 0

        def interrupt():
            nonlocal steps
            steps += 1
            return steps == stop

        db.set_progress_handler(interrupt, 1)
        succeeded = False
        try:
            db.execute("select v,w from t").fetchall()
            succeeded = True
        except sqlite3.OperationalError:
            pass  # Existing readers may map nested interrupts to SQLITE_ERROR.
        db.set_progress_handler(None, 0)
        want = range(1, 20) if db.in_transaction else [1]
        assert db.execute("select rowid,v,w from t order by rowid").fetchall() == [
            (i, vector(i), vector(-i)) for i in want]
        db.close()
        if succeeded:
            break
    else:
        pytest.fail("read did not finish within 1200 SQLite steps")
