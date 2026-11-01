import random
import sqlite3
from collections import deque

import pytest


def connect(path=":memory:"):
    db = sqlite3.connect(path, isolation_level=None)
    db.enable_load_extension(True)
    db.load_extension("dist/vec0")
    db.enable_load_extension(False)
    return db


@pytest.fixture()
def db():
    db = connect()
    db.execute("create table edges(src integer, dst integer)")
    db.executemany(
        "insert into edges values (?, ?)",
        [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (6, 7)],
    )
    return db


def bfs(db, start, max_depth=None, table="edges"):
    rows = db.execute(
        "select node, depth from graph_bfs(?, 'src', 'dst', ?, ?)",
        [table, start, max_depth],
    ).fetchall()
    return {n: d for n, d in rows}


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


def test_bfs_basic(db):
    assert bfs(db, 1) == {1: 0, 2: 1, 3: 1, 4: 2, 5: 3}
    assert bfs(db, 1, 1) == {1: 0, 2: 1, 3: 1}
    assert bfs(db, 1, 0) == {1: 0}
    # unknown and sink nodes return just the start
    assert bfs(db, 99) == {99: 0}
    assert bfs(db, 5) == {5: 0}
    # each node once, in nondecreasing depth order
    depths = [d for _, d in db.execute("select node, depth from graph_bfs('edges', 'src', 'dst', 1)")]
    assert depths == sorted(depths)


def test_bfs_matches_reference_random():
    db = connect()
    rng = random.Random(7)
    edges = [(rng.randrange(500), rng.randrange(500)) for _ in range(3000)]
    db.execute("create table e(a, b)")
    db.executemany("insert into e values (?, ?)", edges)
    for start in range(0, 500, 37):
        for md in (None, 1, 2, 4):
            got = dict(db.execute("select node, depth from graph_bfs('e', 'a', 'b', ?, ?)", [start, md]).fetchall())
            assert got == ref_bfs(edges, start, md)


def test_undirected_view(db):
    db.execute("create view und as select src, dst from edges union all select dst, src from edges")
    assert bfs(db, 5, table="und") == {5: 0, 4: 1, 2: 2, 3: 2, 1: 3}


def test_shortest_path(db):
    sp = lambda a, b: [n for _, n in db.execute(
        "select step, node from graph_shortest_path('edges', 'src', 'dst', ?, ?) order by step", [a, b])]
    p = sp(1, 5)
    assert p[0] == 1 and p[-1] == 5 and len(p) == 4 and p[2] == 4
    assert sp(1, 1) == [1]
    assert sp(5, 1) == []
    assert sp(1, 7) == []
    assert sp(1, 99) == []
    with pytest.raises(sqlite3.OperationalError):
        db.execute("select * from graph_shortest_path('edges', 'src', 'dst', 1)").fetchall()


def test_identifiers_are_quoted(db):
    db.execute('create table "we""ird ; table"("s r c", "d-s-t")')
    db.execute('insert into "we""ird ; table" values (1, 2)')
    rows = db.execute("select node from graph_bfs('we\"ird ; table', 's r c', 'd-s-t', 1)").fetchall()
    assert rows == [(1,), (2,)]
    # injection attempts are just missing names
    for t, s in [("edges; drop table edges", "src"), ("edges", 'src" from edges; --')]:
        with pytest.raises(sqlite3.OperationalError, match="cannot read edges"):
            db.execute("select * from graph_bfs(?, ?, 'dst', 1)", [t, s]).fetchall()
    db.execute("create table `b``q`(src, dst)")
    db.execute("insert into `b``q` values (1, 9)")
    assert bfs(db, 1, table="b`q") == {1: 0, 9: 1}
    assert db.execute("select count(*) from edges").fetchone() == (6,)


def test_errors(db):
    with pytest.raises(sqlite3.OperationalError, match="cannot read edges"):
        db.execute("select * from graph_bfs('nope', 'src', 'dst', 1)").fetchall()
    with pytest.raises(sqlite3.OperationalError, match="cannot read edges"):
        db.execute("select * from graph_bfs('edges', 'src', 'nope', 1)").fetchall()
    with pytest.raises(sqlite3.OperationalError, match="start must be an integer"):
        db.execute("select * from graph_bfs('edges', 'src', 'dst', 'x')").fetchall()
    with pytest.raises(sqlite3.OperationalError, match="max_depth must be >= 0"):
        db.execute("select * from graph_bfs('edges', 'src', 'dst', 1, -1)").fetchall()
    with pytest.raises(sqlite3.OperationalError):
        db.execute("select * from graph_bfs('edges', 'src', 'dst')").fetchall()
    db.execute("insert into edges values (10, 'eleven')")
    with pytest.raises(sqlite3.OperationalError, match="must be integers, got TEXT"):
        db.execute("select * from graph_bfs('edges', 'src', 'dst', 1)").fetchall()


def test_null_endpoints_skipped(db):
    db.execute("insert into edges values (5, null), (null, 1)")
    assert bfs(db, 1) == {1: 0, 2: 1, 3: 1, 4: 2, 5: 3}


def test_autocommit_edits_visible(db):
    assert 6 not in bfs(db, 1)
    db.execute("insert into edges values (5, 6)")
    assert bfs(db, 1)[7] == 5
    db.execute("delete from edges where src = 4")
    assert bfs(db, 1) == {1: 0, 2: 1, 3: 1, 4: 2}
    db.execute("update edges set dst = 5 where src = 1 and dst = 2")
    assert bfs(db, 1) == {1: 0, 5: 1, 3: 1, 4: 2, 6: 2, 7: 3}


def test_transaction_own_writes_and_rollback(db):
    bfs(db, 1)  # warm cache
    db.execute("begin")
    db.execute("insert into edges values (5, 6)")
    assert 7 in bfs(db, 1)  # own uncommitted write is visible
    db.execute("savepoint s")
    db.execute("insert into edges values (7, 8)")
    assert 8 in bfs(db, 1)
    db.execute("rollback to s")
    got = bfs(db, 1)
    assert 8 not in got and 7 in got
    db.execute("rollback")
    assert bfs(db, 1) == {1: 0, 2: 1, 3: 1, 4: 2, 5: 3}
    db.execute("begin")
    db.execute("insert into edges values (5, 6)")
    db.execute("commit")
    assert 7 in bfs(db, 1)


def test_failed_statement_rolls_back(db):
    db.execute("create table g(a integer, b integer check (b < 100))")
    db.execute("insert into g values (1, 2)")
    q = "select node from graph_bfs('g', 'a', 'b', 1)"
    db.execute("begin")
    with pytest.raises(sqlite3.IntegrityError):
        db.executemany("insert into g values (?, ?)", [(2, 3), (3, 500)])
    db.execute("commit")
    assert {r[0] for r in db.execute(q)} == {1, 2, 3}


def test_read_inside_select_with_writes_elsewhere(db):
    # A graph read inside a read transaction stays cached and correct.
    db.execute("begin")
    assert bfs(db, 1)[5] == 3
    assert bfs(db, 1)[5] == 3
    db.execute("commit")


def test_other_connection_commit_visible(tmp_path):
    path = str(tmp_path / "g.db")
    a, b = connect(path), connect(path)
    a.execute("create table edges(src, dst)")
    a.execute("insert into edges values (1, 2)")
    assert bfs(b, 1) == {1: 0, 2: 1}
    a.execute("insert into edges values (2, 3)")
    assert bfs(b, 1) == {1: 0, 2: 1, 3: 2}
    a.execute("drop table edges")
    a.execute("create table edges(src, dst)")
    assert bfs(b, 1) == {1: 0}


def test_more_graphs_than_cache_slots(db):
    for i in range(20):
        db.execute(f"create table e{i}(src, dst)")
        db.execute(f"insert into e{i} values (1, {i + 100})")
    for _ in range(2):
        for i in range(20):
            assert bfs(db, 1, table=f"e{i}") == {1: 0, i + 100: 1}


def test_join_two_traversals(db):
    rows = db.execute(
        """select a.node, b.node from graph_bfs('edges', 'src', 'dst', 1, 1) a
           join graph_bfs('edges', 'src', 'dst', a.node, 1) b
           order by 1, 2"""
    ).fetchall()
    assert rows == [(1, 1), (1, 2), (1, 3), (2, 2), (2, 4), (3, 3), (3, 4)]


def witness(db, source=1, target=5, work=100000, memory=65536):
    import json
    return json.loads(db.execute(
        "select graph_witness('evidence_edges','src','dst','id',?,?,?,?)",
        [source, target, work, memory]).fetchone()[0])


def test_graph_witness_oracle_and_rollback(db):
    db.execute("create table evidence_edges(id integer primary key, src, dst)")
    edges = [(10,1,2),(11,1,3),(12,2,4),(13,3,4),(14,4,5),(15,1,2)]
    db.executemany("insert into evidence_edges values (?,?,?)", edges)
    def check():
        for source in (1, 2, 5, 99):
            for target in (1, 4, 5, 99):
                path = witness(db, source, target)
                distances = ref_bfs([(s,d) for _,s,d in edges], source)
                if target not in distances:
                    assert path == []
                    continue
                assert len(path) == distances[target] + 1
                assert path[0] == [source, None]
                assert path[-1][0] == target
                for previous, current in zip(path, path[1:]):
                    assert (current[1], previous[0], current[0]) in edges
    check()
    db.execute("begin")
    db.execute("savepoint s")
    db.execute("insert into evidence_edges values (20,1,5)")
    assert witness(db) == [[1,None],[5,20]]
    with pytest.raises(sqlite3.Error, match="work limit"):
        witness(db, work=1)
    assert db.in_transaction
    db.execute("rollback to s")
    check()
    db.execute("rollback")
    check()


def test_graph_witness_failure_recovery(db):
    db.execute("create table evidence_edges(id,src,dst)")
    db.executemany("insert into evidence_edges values (?,?,?)", [(1,1,2),(2,2,5)])
    for _ in range(3):
        with pytest.raises(sqlite3.Error, match="memory limit"):
            witness(db, memory=128)
        assert witness(db) == [[1,None],[2,1],[5,2]]
    db.execute("insert into evidence_edges values (1,8,9)")
    with pytest.raises(sqlite3.Error, match="duplicate edge ID"):
        witness(db)
    db.execute("delete from evidence_edges where src=8")
    db.execute("insert into evidence_edges values (NULL,8,9)")
    with pytest.raises(sqlite3.Error, match="non-NULL integers"):
        witness(db)
    db.execute("delete from evidence_edges where src=8")
    assert witness(db)[-1] == [5,2]


def test_graph_witness_cancellation(db):
    import threading
    db.execute("create table evidence_edges(id,src,dst)")
    db.executemany("insert into evidence_edges values (?,?,?)", [(i,i,i+1) for i in range(40000)])
    timer = threading.Timer(0.02, db.interrupt)
    timer.start()
    try:
        with pytest.raises(sqlite3.OperationalError, match="interrupt"):
            witness(db, target=40000, work=10**12, memory=4*1024*1024)
    finally:
        timer.cancel()
        timer.join()
    assert db.execute("select count(*) from evidence_edges").fetchone()[0] == 40000
    assert witness(db, source=7, target=7, work=10**12, memory=4*1024*1024) == [[7,None]]


def test_graph_witness_g0_oracle(db):
    import importlib.util
    import json
    from pathlib import Path
    root = Path('examples/search-lab')
    spec = importlib.util.spec_from_file_location('g0_oracle', root / 'graph_oracle.py')
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    db.execute('create table evidence_edges(id,src,dst)')
    for graph in json.loads((root / 'graph-fixtures.json').read_text())['graphs']:
        edges = graph['edges']
        # Bounded evidence deliberately rejects NULL endpoints; legacy BFS skips them.
        if any(s is None or d is None for _,s,d in edges):
            continue
        db.execute('delete from evidence_edges')
        db.executemany('insert into evidence_edges values (?,?,?)', edges)
        for case in graph['cases']:
            source = case['start']
            distances = oracle.minimum_hops([(s,d) for _,s,d in edges], [source])
            for target in graph['nodes'] + [source]:
                path = witness(db, source, target)
                if target not in distances:
                    assert path == []
                    continue
                assert len(path) == distances[target] + 1
                assert path[0] == [source, None]
                assert path[-1][0] == target
                for a,b in zip(path,path[1:]):
                    assert [b[1],a[0],b[0]] in edges
