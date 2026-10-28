"""G0 conformance fixtures; neither a GraphIR binder nor a new traversal API."""

import asyncio
import json
from pathlib import Path
import random
import sqlite3

import pytest

from rust_test_client import RustAdapter, command
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from fixture import create
from graph_oracle import minimum_hops


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIXTURES = json.loads((HERE / "graph-fixtures.json").read_text())
GRAPHS = FIXTURES["graphs"]
EXTENSION = REPO / "dist" / ("vec0.dylib" if (REPO / "dist/vec0.dylib").exists() else "vec0.so")


def connect(path=":memory:"):
    db = sqlite3.connect(path, isolation_level=None)
    db.enable_load_extension(True)
    db.load_extension(str(EXTENSION))
    db.enable_load_extension(False)
    return db


def load(db, graph):
    db.executescript("create table nodes(id integer primary key); create table edges(id integer primary key,src,dst);")
    db.executemany("insert into nodes values (?)", ((v,) for v in graph["nodes"]))
    db.executemany("insert into edges values (?,?,?)", graph["edges"])


def pairs(graph):
    return [(s,d) for _,s,d in graph["edges"]]


def native(db, start, depth=None, table="edges", src="src", dst="dst"):
    return db.execute("select node,depth from graph_bfs(?,?,?,?,?) order by depth,node",
                      (table,src,dst,start,depth)).fetchall()


def canonical(result):
    return sorted(result.items(),key=lambda r:(r[1],r[0]))


@pytest.mark.parametrize("graph",GRAPHS,ids=lambda g:g["name"])
def test_handwritten_expectations(graph):
    db=connect()
    try:
        load(db,graph)
        for case in graph["cases"]:
            expected=[tuple(row) for row in case["expected"]]
            oracle=minimum_hops(pairs(graph),[case["start"]],case["depth"])
            assert canonical(oracle)==expected
            assert native(db,case["start"],case["depth"])==expected
    finally:
        db.close()


@pytest.mark.parametrize("graph",GRAPHS,ids=lambda g:g["name"])
def test_all_anchors_depths_and_direction_against_oracle(graph):
    db=connect()
    try:
        load(db,graph)
        # Includes sinks, isolated and absent IDs, not just one convenient root.
        for start in graph["nodes"]+[99]:
            for depth in (None,0,1,2,len(graph["nodes"])+1):
                assert native(db,start,depth)==canonical(minimum_hops(pairs(graph),[start],depth))
                reverse=[(d,s) for s,d in pairs(graph)]
                assert native(db,start,depth,src="dst",dst="src")==canonical(minimum_hops(reverse,[start],depth))
        # UNION ALL admits parallel/reverse duplicates; reachability is still a set.
        db.execute("create view undirected as select src,dst from edges union all select dst,src from edges")
        both=pairs(graph)+[(d,s) for s,d in pairs(graph)]
        assert native(db,graph["nodes"][0],table="undirected")==canonical(minimum_hops(both,[graph["nodes"][0]]))
    finally:
        db.close()


@pytest.mark.parametrize("graph",GRAPHS,ids=lambda g:g["name"])
def test_one_shortest_sequence_not_edge_witnesses(graph):
    db=connect()
    try:
        load(db,graph)
        edges=set(pairs(graph))
        for source in graph["nodes"]+[99]:
            distances=minimum_hops(pairs(graph),[source])
            for target in graph["nodes"]+[99]:
                rows=db.execute("select step,node from graph_shortest_path('edges','src','dst',?,?) order by step",
                                (source,target)).fetchall()
                if target not in distances:
                    assert rows==[]
                    continue
                assert [step for step,_ in rows]==list(range(distances[target]+1))
                sequence=[node for _,node in rows]
                assert sequence[0]==source and sequence[-1]==target
                assert len(set(sequence))==len(sequence)
                assert all(pair in edges for pair in zip(sequence,sequence[1:]))
    finally:
        db.close()


def test_minimum_distance_is_not_exact_length_path_existence():
    graph=next(g for g in GRAPHS if g["name"]=="shortcut")
    db=connect()
    try:
        load(db,graph)
        # A two-edge binding exists although the target's minimum distance is one.
        witnesses=db.execute("select a.src,a.dst,b.dst from edges a join edges b on a.dst=b.src where a.src=1").fetchall()
        assert witnesses==[(1,2,3)]
        assert db.execute("select node from graph_bfs('edges','src','dst',1,2) where depth=2").fetchall()==[]
        assert minimum_hops(pairs(graph),[1])[3]==1
    finally:
        db.close()


def test_parallel_edge_bags_optional_nulls_and_node_counts():
    graph=GRAPHS[0]
    db=connect()
    try:
        load(db,graph)
        # SQL edge bindings retain identity and multiplicity, unlike BFS nodes.
        assert db.execute("select id from edges where src=1 and dst=2 order by id").fetchall()==[(101,),(105,)]
        assert db.execute("select count(*) from edges where src=1").fetchone()==(3,)
        assert db.execute("select count(*) from graph_bfs('edges','src','dst',1,1) where depth=1").fetchone()==(2,)
        optional=db.execute("select n.id,e.id,e.dst from nodes n left join edges e on e.src=n.id where n.id=5").fetchall()
        assert optional==[(5,None,None)]
        assert db.execute("select count(*),count(e.id) from nodes n left join edges e on e.src=n.id where n.id=5").fetchone()==(1,0)
        # A nullable property is different from an absent node binding.
        db.execute("alter table nodes add column name text")
        assert db.execute("select id,name,name=null,name is null from nodes where id=5").fetchall()==[(5,None,None,1)]
        assert db.execute("select id from nodes where id=99").fetchall()==[]
    finally:
        db.close()


def test_repeated_variable_requires_equal_node_identity():
    graph=next(g for g in GRAPHS if g["name"]=="cycles")
    db=connect()
    try:
        load(db,graph)
        assert db.execute("select id,src from edges where src=dst order by id").fetchall()==[(101,1),(105,3)]
        assert native(db,1,0)==[(1,0)]
    finally:
        db.close()


def test_labels_are_filtered_edge_sources_not_path_patterns():
    graph=GRAPHS[0]
    db=connect()
    try:
        load(db,graph)
        db.execute("alter table edges add column label text default 'CALLS'")
        db.execute("update edges set label='FLOWS_TO' where id=102")
        db.execute("create view calls as select src,dst from edges where label='CALLS'")
        calls=[(s,d) for edge,s,d in graph["edges"] if edge!=102]
        assert native(db,1,table="calls")==[(1,0),(2,1),(4,2)]
        assert native(db,1,table="calls")==canonical(minimum_hops(calls,[1]))
        assert native(db,1)==[(1,0),(2,1),(3,1),(4,2)]
    finally:
        db.close()


def test_native_null_and_integer_validation_order():
    db=connect()
    try:
        db.execute("create table edges(src,dst)")
        db.executemany("insert into edges values (?,?)",[(1,2),(None,"ignored"),(2,None)])
        assert native(db,1)==[(1,0),(2,1)]
        # The source is checked first; NULL destination does not mask bad source.
        db.execute("insert into edges values ('invalid',null)")
        with pytest.raises(sqlite3.OperationalError,match="must be integers, got TEXT"):
            native(db,1,0)
        db.execute("delete from edges where src='invalid'")
        for start,depth in [(None,0),(1,-1),(1,1.5)]:
            with pytest.raises(sqlite3.OperationalError):
                native(db,start,depth)
    finally:
        db.close()


def test_identity_includes_graph_and_node_type():
    identities=FIXTURES["identity_examples"]
    db=connect()
    try:
        db.execute("create table catalog(graph text,node_type text,id integer,primary key(graph,node_type,id))")
        db.executemany("insert into catalog values (?,?,?)",identities)
        assert len({tuple(v) for v in identities})==3
        assert db.execute("select count(*) from catalog where id=1").fetchone()==(3,)
        assert db.execute("select count(*) from catalog where graph='diamond' and node_type='Function' and id=1").fetchone()==(1,)
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("insert into catalog values (?,?,?)",identities[0])
        # Native table/column selection supplies a domain; it has no type binder.
        db.executescript("create table function_edges(src,dst); create table file_edges(src,dst);")
        db.execute("insert into function_edges values (1,2)")
        db.execute("insert into file_edges values (1,3)")
        assert native(db,1,table="function_edges")==[(1,0),(2,1)]
        assert native(db,1,table="file_edges")==[(1,0),(3,1)]
        edge_keys={(g["name"],edge_id) for g in GRAPHS for edge_id,_,_ in g["edges"]}
        assert len(edge_keys)==sum(len(g["edges"]) for g in GRAPHS)
        assert ("diamond",101) in edge_keys and ("cycles",101) in edge_keys
    finally:
        db.close()


def test_fixed_seed_graphs_and_metamorphic_invariants():
    rng=random.Random(736)
    for _ in range(24):
        vertices=[-9,0,2,7,17,23]
        edges=[(rng.choice(vertices),rng.choice(vertices)) for _ in range(16)]
        db=connect()
        try:
            db.execute("create table edges(id integer primary key,src,dst)")
            for variant in (edges,list(reversed(edges)),edges+edges+[(v,v) for v in vertices]):
                db.execute("delete from edges")
                db.executemany("insert into edges(src,dst) values (?,?)",variant)
                for start in vertices+[99]:
                    for depth in (None,0,1,3):
                        got=native(db,start,depth)
                        assert got==canonical(minimum_hops(edges,[start],depth))
                        assert len({node for node,_ in got})==len(got)
        finally:
            db.close()


def test_write_savepoint_and_transaction_rollback_match_oracle():
    db=connect()
    try:
        load(db,GRAPHS[0])
        def check():
            edges=db.execute("select src,dst from edges").fetchall()
            for start in (1,4,5,99):
                assert native(db,start)==canonical(minimum_hops(edges,[start]))
        check()  # warm the native cache
        db.execute("begin")
        db.execute("insert into edges values (200,4,5)")
        check()
        assert dict(native(db,1))[5]==3
        db.execute("savepoint s")
        db.execute("delete from edges where dst=4")
        check()
        assert 5 not in dict(native(db,1))
        db.execute("rollback to s")
        db.execute("release s")
        check()
        assert dict(native(db,1))[5]==3
        db.execute("rollback")
        check()
        assert 5 not in dict(native(db,1))
        db.execute("begin")
        db.execute("insert into edges values (200,4,5)")
        db.execute("commit")
        check()
        assert dict(native(db,1))[5]==3
    finally:
        db.close()


def test_adapter_catalog_scope_multiseed_and_paging(tmp_path):
    config=create(tmp_path,EXTENSION)
    cfg=json.loads(config.read_text())
    writer=sqlite3.connect(cfg["database"],isolation_level=None)
    # The only route to eligible isolated node 5 uses ineligible node 6.
    writer.executemany("insert into edges values (?,?)",[(1,6),(6,5)])
    writer.close()
    db=connect(cfg["database"])
    adapter=RustAdapter(config)
    try:
        eligible={1,2,3,4,5,10}
        raw=db.execute("select src,dst from edges").fetchall()
        scoped=[(s,d) for s,d in raw if s in eligible and d in eligible]
        assert 5 in minimum_hops(raw,[1],2)
        assert 5 not in minimum_hops(scoped,[1],2)
        assert canonical(minimum_hops(scoped,[1],2))==native(db,1,2,table="admitted_edges")
        for seeds in ([1],[1,2],[5],[99],[6],[1,5,6,99]):
            for depth in (0,1,2,8):
                expected=canonical(minimum_hops(scoped,[s for s in seeds if s in eligible],depth))
                for limit in (1,100):
                    result=adapter.call(dict(op="traverse",corpus="demo",args=dict(ids=seeds,depth=depth,limit=limit)))
                    assert result["ok"],result
                    assert result["rows"]==[dict(id=n,depth=d) for n,d in expected[:limit]]
                    assert result["missing_ids"]==[s for s in seeds if s not in eligible]
                    assert result["total_reachable"]==len(expected)
                    assert result["truncated"]==(len(expected)>limit)
                    assert result["complete"]==(len(expected)<=limit)
        # Native absent anchor behavior remains distinct from catalog admission.
        assert native(db,99)==[(99,0)]
        adapter.close()
        cfg["corpora"]["demo"]["edges"]="edges"
        config.write_text(json.dumps(cfg));adapter=RustAdapter(config)
        assert adapter.call(dict(op="traverse",corpus="demo",args=dict(ids=[1])))["error"]["code"]=="graph_scope"
    finally:
        adapter.close()
        db.close()


@pytest.mark.parametrize("graph",GRAPHS,ids=lambda g:g["name"])
def test_rust_cli_and_mcp_cover_g0_fixtures(graph,tmp_path):
    config=create(tmp_path,EXTENSION)
    cfg=json.loads(config.read_text())
    db=connect(cfg["database"])
    try:
        db.execute("delete from edges")
        db.execute("delete from documents")
        db.execute("delete from lexical")
        for key in range(1,11):
            db.execute("delete from vectors where rowid=?",(key,))
        for key in graph["nodes"]:
            db.execute("insert into documents(id,document_id,tenant,valid_from,source_id,body) values (?,?,'ours',0,?,'evidence')",
                       (key,f"doc:{key}",f"source:{key}"))
            db.execute("insert into vectors(rowid,embedding) values (?,'[0,0]')",(key,))
            db.execute("insert into lexical(rowid,body) values (?,'evidence')",(key,))
        db.executemany("insert into edges values (?,?)",pairs(graph))
    finally:
        db.close()
    def check(result,case):
        admitted=case["start"] in graph["nodes"]
        expected=case["expected"] if admitted else []
        assert result["ok"],result
        assert result["rows"]==[dict(id=n,depth=d) for n,d in expected]
        assert result["missing_ids"]==([] if admitted else [case["start"]])
        assert result["total_reachable"]==len(expected)
    def args(case):
        return dict(ids=[case["start"]],depth=8 if case["depth"] is None else case["depth"],limit=100)
    client=RustAdapter(config)
    try:
        for case in graph["cases"]:
            check(client.call(dict(op="traverse",corpus="demo",args=args(case))),case)
    finally:
        client.close()
    async def run():
        cmd=command(config,"mcp")
        async with stdio_client(StdioServerParameters(command=cmd[0],args=cmd[1:])) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize()
                for case in graph["cases"]:
                    result=await session.call_tool("traverse",dict(corpus="demo",args=args(case)))
                    assert not result.isError
                    check(result.structuredContent,case)
    asyncio.run(run())
