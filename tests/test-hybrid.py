"""The hybrid extension: vec0 and lance0 in one library and one database.

Build first with `make hybrid` (needs Rust and protoc); skipped otherwise.
"""
import glob
import os
import random
import re
import shutil
import sqlite3
import struct

import pytest

EXT = "dist/vec0_lance"
pytestmark = pytest.mark.skipif(
    not glob.glob(EXT + ".*"), reason="hybrid extension not built (make hybrid)"
)


def _f32(v):
    return struct.pack(f"{len(v)}f", *v)


def connect(path):
    db = sqlite3.connect(path, isolation_level=None)  # explicit transactions only
    db.enable_load_extension(True)
    db.load_extension(EXT)
    db.enable_load_extension(False)
    return db


@pytest.fixture()
def root(tmp_path):
    return str(tmp_path)


@pytest.fixture()
def db(root):
    return connect(os.path.join(root, "app.db"))


def lance_table(db, root, name="docs", dim=4):
    db.execute(f"create virtual table {name} using lance0('{root}/{name}.lance', {dim})")


def test_one_load_registers_everything(db):
    modules = {r[0] for r in db.execute("select name from pragma_module_list")}
    assert {"vec0", "vec_each", "vec_sparse_each", "lance0"} <= modules
    functions = {r[0] for r in db.execute("select name from pragma_function_list")}
    assert {"vec_version", "vec_f32", "vec_distance_l2", "vec_rrf", "vec_sparse"} <= functions


# ---------------------------------------------------------------------------
# One vector format
# ---------------------------------------------------------------------------


def test_vectors_round_trip_between_vec0_and_lance0(db, root):
    lance_table(db, root)
    db.execute("create virtual table items using vec0(embedding float[4])")
    db.execute("insert into items(rowid, embedding) values (1, '[0.5, -1, 2, 3.25]')")
    # vec0's stored vector goes into lance0 unchanged...
    db.execute("insert into docs(rowid, label, vector) select rowid, 'a', embedding from items")
    # ...and JSON text works like it does in vec0
    db.execute("insert into docs(rowid, label, vector) values (2, 'b', '[1, 2, 3, 4]')")
    blob = db.execute("select vector from docs where rowid = 1").fetchone()[0]
    assert blob == db.execute("select embedding from items where rowid = 1").fetchone()[0]
    assert db.execute("select vec_to_json(vector) from docs where rowid = 2").fetchone()[0] == \
        "[1.000000,2.000000,3.000000,4.000000]"
    assert db.execute("""select vec_distance_l2(d.vector, i.embedding)
                         from docs d join items i on i.rowid = d.rowid""").fetchone()[0] == 0.0
    for bad in ["[1, 2]", "not json", "[1, 2, x, 4]"]:
        with pytest.raises(sqlite3.Error, match="lance0"):
            db.execute("insert into docs(rowid, vector) values (9, ?)", [bad])


def test_lance0_knn_matches_vec0(db, root):
    rng = random.Random(0)
    rows = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(300)]
    lance_table(db, root, dim=8)
    db.execute("create virtual table items using vec0(embedding float[8])")
    db.execute("begin")
    for i, v in enumerate(rows):
        db.execute("insert into items(rowid, embedding) values (?, ?)", [i, _f32(v)])
        db.execute("insert into docs(rowid, vector) values (?, ?)", [i, _f32(v)])
    db.execute("commit")
    for q in rows[:10]:
        a = db.execute("select rowid, distance from items where embedding match ? and k = 5",
                       [_f32(q)]).fetchall()
        b = db.execute("select rowid, distance from docs where query = ? and k = 5 order by distance",
                       [_f32(q)]).fetchall()
        assert [r[0] for r in a] == [r[0] for r in b]
        # same units: both report L2, not squared L2
        assert [r[1] for r in b] == pytest.approx([r[1] for r in a], rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# One query across both engines
# ---------------------------------------------------------------------------


def test_join_vec0_index_lance0_payloads_and_tables(db, root):
    """vec0 HNSW finds neighbors; lance0 holds their blobs; SQLite the rest."""
    lance_table(db, root, dim=2)
    db.execute("create table users(id integer primary key, name text)")
    db.execute("create virtual table emb using vec0(v float[2] indexed by hnsw())")
    db.execute("begin")
    for i in range(1, 21):
        db.execute("insert into users values (?, ?)", [i, f"user{i}"])
        db.execute("insert into emb(rowid, v) values (?, ?)", [i, _f32([i, 0])])
        db.execute("insert into docs(rowid, label, vector, payload) values (?, ?, ?, ?)",
                   [i, "img", _f32([i, 0]), bytes([i]) * (1000 * i)])
    db.execute("commit")
    got = db.execute("""
        select u.name, length(d.payload), e.distance
        from emb e
        join docs d on d.rowid = e.rowid
        join users u on u.id = e.rowid
        where e.v match ? and e.k = 3
        order by e.distance""", [_f32([7.2, 0])]).fetchall()
    assert [(n, l) for n, l, _ in got] == [("user7", 7000), ("user8", 8000), ("user6", 6000)]

    # and the other way round: Lance KNN, filtered in Lance, joined to SQLite
    got = db.execute("""
        select u.name from docs d join users u on u.id = d.rowid
        where d.query = ? and d.k = 2 and d.label = 'img' order by d.distance""",
                     [_f32([15.1, 0])]).fetchall()
    assert got == [("user15",), ("user16",)]


# ---------------------------------------------------------------------------
# One transaction across both engines
# ---------------------------------------------------------------------------


def write_all(db, rowid, x):
    db.execute("insert into users values (?, ?)", [rowid, f"user{rowid}"])
    db.execute("insert into emb(rowid, v) values (?, ?)", [rowid, _f32([x, 0])])
    db.execute("insert into docs(rowid, label, vector) values (?, 'x', ?)", [rowid, _f32([x, 0])])


def counts(db):
    return tuple(db.execute(f"select count(*) from {t}").fetchone()[0] for t in ("users", "emb", "docs"))


@pytest.fixture()
def three(db, root):
    lance_table(db, root, dim=2)
    db.execute("create table users(id integer primary key, name text)")
    db.execute("create virtual table emb using vec0(v float[2] indexed by hnsw())")
    db.execute("begin")
    for i in range(1, 6):
        write_all(db, i, i)
    db.execute("commit")
    return db


def test_commit_and_rollback_cover_both(three, root):
    db = three
    db.execute("begin")
    write_all(db, 6, 6)
    db.execute("rollback")
    assert counts(db) == (5, 5, 5)
    db.execute("begin")
    write_all(db, 6, 6)
    db.execute("commit")
    db.close()
    db = connect(os.path.join(root, "app.db"))
    assert counts(db) == (6, 6, 6)
    assert db.execute("select rowid from docs where query = ? and k = 1", [_f32([6, 0])]).fetchone() == (6,)


def test_reads_see_own_writes_in_a_transaction(three):
    db = three
    db.execute("begin")
    db.execute("insert into docs(rowid, label, vector) values (50, 'new', ?)", [_f32([50, 0])])
    db.execute("delete from docs where rowid = 2")
    db.execute("update docs set label = 'moved', vector = ? where rowid = 3", [_f32([49, 0])])
    assert db.execute("select label from docs where rowid = 50").fetchone() == ("new",)
    assert db.execute("select count(*) from docs where rowid = 2").fetchone() == (0,)
    knn = db.execute("select rowid, label from docs where query = ? and k = 3 order by distance",
                     [_f32([49.4, 0])]).fetchall()
    assert knn == [(3, "moved"), (50, "new"), (5, "x")]
    # insert then delete in one transaction leaves nothing behind
    db.execute("insert into docs(rowid, vector) values (60, ?)", [_f32([60, 0])])
    db.execute("delete from docs where rowid = 60")
    db.execute("commit")
    assert sorted(r[0] for r in db.execute("select rowid from docs")) == [1, 3, 4, 5, 50]
    assert db.execute("select label from docs where rowid = 3").fetchone() == ("moved",)


def test_failed_statement_rolls_back_in_both(three):
    db = three
    db.execute("create table src(id, v)")
    db.executemany("insert into src values (?, ?)",
                   [(10, _f32([10, 0])), (11, _f32([11, 0])), (12, _f32([12]))])  # last is wrong
    db.execute("begin")
    db.execute("insert into docs(rowid, vector) values (9, ?)", [_f32([9, 0])])
    with pytest.raises(sqlite3.Error):
        db.execute("insert into docs(rowid, vector) select id, v from src order by id")
    with pytest.raises(sqlite3.Error):
        db.execute("insert into emb(rowid, v) select id, v from src order by id")
    assert db.execute("select count(*) from docs where rowid >= 10").fetchone() == (0,)
    db.execute("savepoint s")
    db.execute("insert into docs(rowid, vector) values (20, ?)", [_f32([20, 0])])
    db.execute("insert into emb(rowid, v) values (20, ?)", [_f32([20, 0])])
    db.execute("rollback to s")
    db.execute("release s")
    db.execute("commit")
    assert sorted(r[0] for r in db.execute("select rowid from docs")) == [1, 2, 3, 4, 5, 9]
    assert db.execute("select count(*) from emb").fetchone() == (5,)


def test_failure_during_commit_leaves_no_orphan(three, root):
    """Lance commits table A, then table B fails: SQLite rolls everything back."""
    db = three
    lance_table(db, root, name="other", dim=2)
    db.execute("begin")
    write_all(db, 7, 7)                                   # docs (A) is written first
    db.execute("insert into other(rowid, vector) values (1, ?)", [_f32([1, 0])])
    shutil.rmtree(os.path.join(root, "other.lance"))     # B can no longer commit:
    open(os.path.join(root, "other.lance"), "w").close()  # its path is now a file
    with pytest.raises(sqlite3.Error):
        db.execute("commit")
    if db.in_transaction:
        db.execute("rollback")
    assert counts(db) == (5, 5, 5)                        # A's Lance write is invisible
    db.execute("drop table other")
    db.execute("begin")
    write_all(db, 8, 8)                                   # the next writer restores A
    db.execute("commit")
    assert counts(db) == (6, 6, 6)
    assert sorted(r[0] for r in db.execute("select rowid from docs")) == [1, 2, 3, 4, 5, 8]


# ---------------------------------------------------------------------------
# vec0 with storage=lance: auxiliary columns in Lance, index in SQLite
# ---------------------------------------------------------------------------


def storage_table(db, root, name="items", index="indexed by hnsw()"):
    db.execute(f"""create virtual table {name} using vec0(
        v float[2] {index},
        +title text, +image blob, +score float, +n integer,
        storage=lance, storage_uri='{root}/{name}.lance')""")


def test_vec0_storage_lance_round_trip(db, root):
    storage_table(db, root)
    names = {r[0] for r in db.execute("select name from sqlite_master")}
    assert "items_auxiliary" not in names and "items_storage" in names
    db.execute("begin")
    for i in range(1, 31):
        db.execute("insert into items(rowid, v, title, image, score, n) values (?, ?, ?, ?, ?, ?)",
                   [i, _f32([i, 0]), f"t{i}", bytes([i]) * 5000, i / 2, i * 10])
    db.execute("insert into items(rowid, v) values (99, ?)", [_f32([99, 0])])  # all NULL
    db.execute("commit")
    # KNN: rowids come from vec0's HNSW, aux values from Lance (one batch)
    got = db.execute("""select rowid, title, length(image), score, n from items
                        where v match ? and k = 3""", [_f32([10.2, 0])]).fetchall()
    assert got == [(10, "t10", 5000, 5.0, 100), (11, "t11", 5000, 5.5, 110), (9, "t9", 5000, 4.5, 90)]
    assert db.execute("select title, image, score, n from items where rowid = 99").fetchone() == \
        (None, None, None, None)
    # full scan and filters on aux values (applied by SQLite)
    assert db.execute("select count(*) from items where n > 200").fetchone() == (10,)
    # update and delete
    db.execute("update items set title = 'renamed', score = 0.25 where rowid = 2")
    db.execute("delete from items where rowid = 3")
    assert db.execute("select title, score, n from items where rowid = 2").fetchone() == ("renamed", 0.25, 20)
    assert db.execute("select count(*) from items").fetchone() == (30,)
    with pytest.raises(sqlite3.Error, match="type mismatch"):
        db.execute("insert into items(rowid, v, n) values (200, ?, 'not a number')", [_f32([0, 0])])


def test_vec0_storage_lance_location_rename_drop(root):
    path = os.path.join(root, "app.db")
    db = connect(path)
    db.execute("create virtual table t using vec0(v float[2], +doc blob, storage=lance)")
    db.execute("insert into t(rowid, v, doc) values (1, ?, x'0102')", [_f32([1, 1])])
    assert os.path.isdir(path + ".t.lance")          # default: next to the database
    db.close()
    db = connect(path)                                # reopened: data persisted
    assert db.execute("select doc from t where rowid = 1").fetchone() == (b"\x01\x02",)
    db.execute("alter table t rename to t2")
    assert db.execute("select doc from t2 where rowid = 1").fetchone() == (b"\x01\x02",)
    db.execute("drop table t2")
    assert not os.path.exists(path + ".t.lance")      # local data removed with the table
    mem = connect(":memory:")
    with pytest.raises(sqlite3.Error, match="storage_uri is required"):
        mem.execute("create virtual table t using vec0(v float[2], +doc blob, storage=lance)")
    with pytest.raises(sqlite3.Error, match="supports flat and hnsw"):
        mem.execute(f"create virtual table t using vec0(v float[8] indexed by rescore(quantizer=bit), "
                    f"storage=lance, storage_uri='{root}/x')")


def test_vec0_storage_lance_transactions(db, root):
    storage_table(db, root)
    db.execute("begin")
    for i in range(1, 6):
        db.execute("insert into items(rowid, v, title) values (?, ?, ?)", [i, _f32([i, 0]), f"t{i}"])
    db.execute("commit")

    # rollback
    db.execute("begin")
    db.execute("insert into items(rowid, v, title) values (6, ?, 'gone')", [_f32([6, 0])])
    db.execute("update items set title = 'changed' where rowid = 1")
    assert db.execute("select title from items where rowid = 1").fetchone() == ("changed",)  # own write
    db.execute("rollback")
    assert db.execute("select title from items where rowid = 1").fetchone() == ("t1",)
    assert db.execute("select count(*) from items").fetchone() == (5,)

    # a failed statement undoes only itself, in SQLite and in Lance
    db.execute("create table src(id, v, title)")
    db.executemany("insert into src values (?, ?, ?)",
                   [(10, _f32([10, 0]), "a"), (11, _f32([11, 0]), "b"), (12, _f32([12]), "bad")])
    db.execute("begin")
    db.execute("insert into items(rowid, v, title) values (9, ?, 'kept')", [_f32([9, 0])])
    with pytest.raises(sqlite3.Error):
        db.execute("insert into items(rowid, v, title) select id, v, title from src order by id")
    db.execute("savepoint s")
    db.execute("update items set title = 'undone' where rowid = 9")
    db.execute("rollback to s")
    db.execute("release s")
    db.execute("commit")
    assert sorted(db.execute("select rowid, title from items")) == \
        [(1, "t1"), (2, "t2"), (3, "t3"), (4, "t4"), (5, "t5"), (9, "kept")]

    # a commit that fails after Lance wrote: nothing becomes visible
    lance_table(db, root, name="other", dim=2)
    db.execute("begin")
    db.execute("insert into items(rowid, v, title) values (20, ?, 'orphan')", [_f32([20, 0])])
    db.execute("insert into other(rowid, vector) values (1, ?)", [_f32([1, 0])])
    shutil.rmtree(os.path.join(root, "other.lance"))
    open(os.path.join(root, "other.lance"), "w").close()
    with pytest.raises(sqlite3.Error):
        db.execute("commit")
    if db.in_transaction:
        db.execute("rollback")
    db.execute("drop table other")
    assert db.execute("select count(*) from items where rowid = 20").fetchone() == (0,)
    db.execute("insert into items(rowid, v, title) values (21, ?, 'after')", [_f32([21, 0])])
    assert db.execute("select title from items where v match ? and k = 1", [_f32([21, 0])]).fetchone() == ("after",)


@pytest.mark.parametrize("index", ["", "indexed by hnsw()"])
def test_vec0_vectors_in_lance(db, root, index):
    """Vectors live in Lance; vec0 (flat scan or HNSW) still answers queries."""
    rng = random.Random(7)
    rows = [[rng.gauss(0, 1) for _ in range(16)] for _ in range(400)]
    db.execute(f"""create virtual table items using vec0(v float[16] {index}, +tag text,
                   storage=lance, storage_uri='{root}/items.lance')""")
    db.execute("create virtual table ref using vec0(v float[16])")
    db.execute("begin")
    for i, v in enumerate(rows):
        db.execute("insert into items(rowid, v, tag) values (?, ?, ?)", [i, _f32(v), f"t{i}"])
        db.execute("insert into ref(rowid, v) values (?, ?)", [i, _f32(v)])
    db.execute("commit")
    names = {r[0] for r in db.execute("select name from sqlite_master")}
    assert not any(n.startswith("items_vector_chunks") for n in names)  # no vectors in SQLite
    if index:
        assert db.execute("select max(length(vector)) from items_hnsw_nodes00").fetchone() == (0,)
    hits = 0
    for q in rows[:20]:
        got = [r[0] for r in db.execute("select rowid from items where v match ? and k = 5", [_f32(q)])]
        want = [r[0] for r in db.execute("select rowid from ref where v match ? and k = 5", [_f32(q)])]
        hits += len(set(got) & set(want))
    assert hits / 100 >= (1.0 if not index else 0.95)
    assert db.execute("select v, tag from items where rowid = 7").fetchone() == (_f32(rows[7]), "t7")
    # filters and distance constraints on the flat path
    got = db.execute("select rowid from items where v match ? and k = 3 and rowid in (1, 2, 3, 4)",
                     [_f32(rows[2])]).fetchall()
    assert got[0] == (2,) and len(got) == 3
    # updates and rollback reach the vectors too
    db.execute("update items set v = ? where rowid = 5", [_f32([50.0] * 16)])
    assert db.execute("select rowid from items where v match ? and k = 1", [_f32([50.0] * 16)]).fetchone() == (5,)
    db.execute("begin")
    db.execute("update items set v = ? where rowid = 6", [_f32([60.0] * 16)])
    db.execute("rollback")
    assert db.execute("select v from items where rowid = 6").fetchone() == (_f32(rows[6]),)
    if index:
        db.execute("insert into items(items) values ('hnsw-rebuild')")
        assert db.execute("select rowid from items where v match ? and k = 1", [_f32(rows[9])]).fetchone() == (9,)


def test_column_patches(db, root):
    """UPDATE of some columns patches only those columns in Lance."""
    lance_table(db, root, dim=4)
    storage_table(db, root)
    db.execute("begin")
    for i in range(1, 6):
        db.execute("insert into docs(rowid, label, vector, payload) values (?, ?, ?, ?)",
                   [i, f"l{i}", _f32([i, 0, 0, 0]), bytes([i]) * 100])
        db.execute("insert into items(rowid, v, title, image, n) values (?, ?, ?, ?, ?)",
                   [i, _f32([i, 0]), f"t{i}", bytes([i]) * 100, i])
    db.execute("commit")
    # one column; the others (vector, blob) must survive untouched
    db.execute("update docs set label = 'L2' where rowid = 2")
    db.execute("update items set title = 'T2', n = 20 where rowid = 2")
    assert db.execute("select label, vector, payload from docs where rowid = 2").fetchone() == \
        ("L2", _f32([2, 0, 0, 0]), bytes([2]) * 100)
    assert db.execute("select title, image, n, v from items where rowid = 2").fetchone() == \
        ("T2", bytes([2]) * 100, 20, _f32([2, 0]))
    # patches accumulate within a transaction, and mix with whole-row writes
    db.execute("begin")
    db.execute("update items set title = 'a' where rowid = 3")
    db.execute("update items set n = 33 where rowid = 3")
    db.execute("update items set v = ? where rowid = 3", [_f32([30, 0])])
    db.execute("insert into items(rowid, v, title) values (9, ?, 'new')", [_f32([9, 0])])
    db.execute("update items set title = 'newer' where rowid = 9")  # still a plain insert
    db.execute("update docs set vector = ?, label = 'moved' where rowid = 4", [_f32([40, 0, 0, 0])])
    db.execute("commit")
    assert db.execute("select title, n, v from items where rowid = 3").fetchone() == ("a", 33, _f32([30, 0]))
    assert db.execute("select title from items where rowid = 9").fetchone() == ("newer",)
    assert db.execute("select rowid from items where v match ? and k = 1", [_f32([30, 0])]).fetchone() == (3,)
    assert db.execute("select label, payload from docs where rowid = 4").fetchone() == ("moved", bytes([4]) * 100)
    assert db.execute("select rowid from docs where query = ? and k = 1", [_f32([40, 0, 0, 0])]).fetchone() == (4,)
    # rolled-back patches leave nothing behind
    db.execute("begin")
    db.execute("update items set title = 'gone' where rowid = 1")
    db.execute("rollback")
    assert db.execute("select title from items where rowid = 1").fetchone() == ("t1",)


def test_compaction_merges_small_fragments(db, root):
    """Many small transactions must not leave a fragment per transaction."""
    lance_table(db, root, dim=4)
    for txn in range(20):
        db.execute("begin")
        for i in range(5):
            rowid = txn * 5 + i + 1
            db.execute("insert into docs(rowid, label, vector) values (?, ?, ?)",
                       [rowid, f"l{rowid}", _f32([rowid, 0, 0, 0])])
        db.execute("commit")
    # Old versions stay on disk for crash recovery, so count the data files
    # the latest manifest references (names are u64::MAX - version).
    manifests = glob.glob(os.path.join(root, "docs.lance", "_versions", "*.manifest"))
    latest = min(manifests, key=lambda f: int(os.path.basename(f).split(".")[0]))
    with open(latest, "rb") as f:
        live = set(re.findall(rb"[0-9a-f]{16,}\.lance", f.read()))
    assert 0 < len(live) < 20
    assert db.execute("select count(*) from docs").fetchone()[0] == 100
    assert db.execute("select label from docs where rowid = 57").fetchone()[0] == "l57"
    # deletes materialize on the next compaction-triggering commit
    db.execute("delete from docs where rowid <= 50")
    for txn in range(20):
        db.execute("begin")
        db.execute("insert into docs(rowid, label, vector) values (?, 'x', ?)",
                   [1000 + txn, _f32([0, 0, 0, 0])])
        db.execute("commit")
    assert db.execute("select count(*) from docs").fetchone()[0] == 70
    assert db.execute("select count(*) from docs where rowid <= 50").fetchone()[0] == 0


@pytest.mark.parametrize("kind", ["lance0", "flat", "hnsw"])
def test_optimize_command(db, root, kind):
    if kind == "lance0":
        lance_table(db, root, dim=2)
        table, shadow, vector = "docs", "docs_lance", "vector"
    else:
        storage_table(db, root, index="" if kind == "flat" else "indexed by hnsw()")
        table, shadow, vector = "items", "items_storage", "v"
    for i in range(1, 5):
        db.execute(f"insert into {table}(rowid, {vector}) values (?, ?)", [i, _f32([i, 0])])

    def version():
        return db.execute(f"select v from {shadow} where k = 'version'").fetchone()[0]

    before = version()
    db.execute("begin")
    db.execute("savepoint s")
    db.execute(f"insert into {table}({table}) values ('optimize')")
    assert version() == before  # maintenance waits for xSync
    db.execute("rollback to s")
    db.execute("release s")
    db.execute("commit")
    assert version() == before

    db.execute("begin")
    db.execute(f"insert into {table}({table}) values ('optimize')")
    db.execute("rollback")
    assert version() == before

    db.execute("begin")
    with pytest.raises(sqlite3.Error, match="unknown.*command"):
        db.execute(f"insert into {table}({table}) select 'optimize' union all select 'unknown'")
    db.execute("commit")
    assert version() == before

    db.execute("begin")
    db.execute(f"insert into {table}({table}) values ('optimize')")
    db.execute("savepoint s")
    db.execute("rollback to s")
    db.execute("release s")
    db.execute("commit")
    assert version() > before
    assert db.execute(f"select v from {shadow} where k = 'unindexed'").fetchone() == (0,)
    assert db.execute(f"select rowid, {vector} from {table} order by rowid").fetchall() == [
        (i, _f32([i, 0])) for i in range(1, 5)
    ]
    for cmd in ["unknown", "cleanup=0"]:
        with pytest.raises(sqlite3.Error, match="unknown.*command"):
            db.execute(f"insert into {table}({table}) values (?)", [cmd])

    # A later table failing xSync must leave the pre-maintenance version readable.
    before = version()
    lance_table(db, root, name="other", dim=2)
    db.execute("begin")
    db.execute(f"insert into {table}({table}) values ('optimize')")
    db.execute("insert into other(rowid, vector) values (1, '[1, 0]')")
    shutil.rmtree(os.path.join(root, "other.lance"))
    open(os.path.join(root, "other.lance"), "w").close()
    with pytest.raises(sqlite3.Error):
        db.execute("commit")
    if db.in_transaction:
        db.execute("rollback")
    assert version() == before
    assert db.execute(f"select count(*) from {table}").fetchone() == (4,)
    db.execute("drop table other")
    db.execute(f"insert into {table}(rowid, {vector}) values (5, '[5, 0]')")
    assert db.execute(f"select {vector} from {table} where rowid = 5").fetchone() == (_f32([5, 0]),)


def test_optimize_compacts_and_preserves_versions(db, root):
    """Manual compaction retains old manifests and leaves queries unchanged."""
    lance_table(db, root, dim=2)
    storage_table(db, root)
    for i in range(1, 13):  # one Lance fragment per transaction
        db.execute("insert into docs(rowid, label, vector) values (?, 'x', ?)", [i, _f32([i, 0])])
        db.execute("insert into items(rowid, v, title) values (?, ?, ?)", [i, _f32([i, 0]), f"t{i}"])
    old_versions = set(glob.glob(os.path.join(root, "items.lance", "_versions", "*.manifest")))
    db.execute("insert into docs(docs) values ('optimize')")
    db.execute("insert into items(items) values ('optimize')")
    manifests = glob.glob(os.path.join(root, "items.lance", "_versions", "*.manifest"))
    assert old_versions <= set(manifests)
    latest = min(manifests, key=lambda f: int(os.path.basename(f).split(".")[0]))
    with open(latest, "rb") as f:
        data = f.read()
    # The prefix also stores index/transaction history with obsolete file names.
    offset = struct.unpack_from("<q", data, len(data) - 16)[0]
    length = struct.unpack_from("<I", data, offset)[0]
    assert data[-4:] == b"LANC" and offset + 4 + length == len(data) - 16
    live = set(re.findall(rb"[0-9a-f]{16,}\.lance", data[offset + 4:offset + 4 + length]))
    assert 0 < len(live) < 12
    assert db.execute("select count(*) from docs").fetchone() == (12,)
    assert db.execute("select rowid from docs where query = ? and k = 1", [_f32([7.2, 0])]).fetchone() == (7,)
    assert sorted(db.execute("select rowid, title from items where rowid in (3, 4)")) == [(3, "t3"), (4, "t4")]
