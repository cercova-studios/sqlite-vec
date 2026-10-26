import asyncio
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from adapter import Adapter, LIMITS, cli
from fixture import create


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXTENSION = REPO / "dist/vec0.dylib"
if not EXTENSION.exists():
    EXTENSION = REPO / "dist/vec0.so"


@pytest.fixture
def config(tmp_path):
    return create(tmp_path, EXTENSION)


@pytest.fixture
def adapter(config):
    a = Adapter(config)
    yield a
    a.close()


def request(op, args=None):
    return dict(op=op, corpus="demo", args=args or {})


def clean(result):
    return {k:v for k,v in result.items() if k != "elapsed_ms"}


def direct(config):
    """Independent equivalent SQL, not the adapter's query templates."""
    cfg = json.loads(config.read_text())
    d = sqlite3.connect(Path(cfg["database"]).as_uri()+"?mode=ro",uri=True)
    d.row_factory = sqlite3.Row
    d.enable_load_extension(True)
    d.load_extension(cfg["extension"])
    d.enable_load_extension(False)
    return d


def oracle(config):
    with direct(config) as d:
        # Exhaustive scalar L2 + window ranks is independent of vec0 KNN.
        rows = d.execute("""
          with eligible as materialized (
            select * from documents where tenant='ours' and invalidated_at is null
              and valid_from<=100 and (valid_to is null or valid_to>100)
          ), vector_raw as (
            select v.rowid id,vec_distance_l2(v.embedding,'[0,0]') distance
            from vectors v join eligible e on e.id=v.rowid
            order by distance,id limit 3
          ), v as (select *,row_number() over(order by distance,id) vr from vector_raw),
          keyword_raw as (
            select rowid id,rank bm25 from lexical where lexical match 'deploy'
              and rowid in (select id from eligible) order by rank,rowid limit 3
          ), f as (select *,row_number() over(order by bm25,id) kr from keyword_raw)
          select coalesce(v.id,f.id) id,vr vector_rank,kr keyword_rank,
            distance vector_distance,bm25 keyword_bm25,
            coalesce(1.0/(60+vr),0)+coalesce(1.0/(60+kr),0) score,e.source_id
          from v full join f on v.id=f.id join eligible e on e.id=coalesce(v.id,f.id)
          order by score desc,id limit 3
        """).fetchall()
        search = [dict(r) for r in rows]
        seeds = [r["id"] for r in rows]
        # Direct native SQL plus a recursive SQL oracle verifies hop semantics.
        native = {}
        for key in seeds:
            for node,depth in d.execute("select node,depth from graph_bfs('admitted_edges','src','dst',?,2)", (key,)):
                native[node] = min(native.get(node,depth),depth)
        rec = d.execute("""
          with recursive walk(node,depth) as (
            select value,0 from json_each(?) union
            select e.dst,w.depth+1 from walk w join admitted_edges e on e.src=w.node where w.depth<2
          ) select node,min(depth) depth from walk group by node order by depth,node
        """, (json.dumps(seeds),)).fetchall()
        assert sorted(native.items(),key=lambda r:(r[1],r[0])) == [tuple(r) for r in rec]
        graph = [dict(id=r[0],depth=r[1]) for r in rec]
        evidence = [dict(d.execute("select id,document_id,body,source_id,start_ms,end_ms from documents where id=?", (r["id"],)).fetchone()) for r in graph]
    return search,graph,evidence


def journey(call, config):
    description = call(request("describe"))
    assert description["ok"] and description["dimensions"] == 2
    assert description["sqlite_version"] == sqlite3.sqlite_version
    assert not description["cancellation"] and not description["approximation"]
    search = call(request("search",dict(text="deploy",vector=[0,0],candidates=3,limit=3)))
    assert search["ok"], search
    graph = call(request("traverse",dict(ids=[r["id"] for r in search["rows"]],depth=2)))
    assert graph["ok"],graph
    evidence = call(request("fetch",dict(ids=[r["id"] for r in graph["rows"]])))
    want = oracle(config)
    assert search["rows"] == want[0]
    assert graph["rows"] == want[1]
    assert evidence["rows"] == want[2]
    assert not evidence["missing_ids"]
    return [description, search, graph, evidence]


def test_cli_journey(config):
    # One real JSONL process, not just calls to a Python helper.
    want_search,want_graph,_ = oracle(config)
    records = [request("describe"), request("search",dict(text="deploy",vector=[0,0],candidates=3,limit=3)),
               request("traverse",dict(ids=[r["id"] for r in want_search],depth=2)),
               request("fetch",dict(ids=[r["id"] for r in want_graph]))]
    p = subprocess.run([sys.executable,str(HERE/"adapter.py"),"--config",str(config),"cli"],
                       input="".join(json.dumps(r)+"\n" for r in records),text=True,capture_output=True,check=True)
    results = iter(json.loads(line) for line in p.stdout.splitlines())
    journey(lambda r:next(results),config)
    assert not list(results)
    assert not p.stderr


def test_handlers_journey(adapter,config):
    journey(adapter.call, config)


def test_mcp_journey_and_errors(config,adapter):
    async def run():
        params = StdioServerParameters(command=sys.executable,args=[str(HERE/"adapter.py"),"--config",str(config),"mcp"])
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert [t.name for t in tools.tools] == ["describe","search","traverse","fetch"]
                assert all(t.annotations.readOnlyHint for t in tools.tools)
                results = []
                for rec in [request("describe"), request("search",dict(text="deploy",vector=[0,0],candidates=3,limit=3))]:
                    out = await session.call_tool(rec["op"],dict(corpus="demo",args=rec["args"]))
                    assert not out.isError
                    results.append(out.structuredContent)
                graph = await session.call_tool("traverse",dict(corpus="demo",args=dict(ids=[r["id"] for r in results[1]["rows"]],depth=2)))
                results.append(graph.structuredContent)
                evidence = await session.call_tool("fetch",dict(corpus="demo",args=dict(ids=[r["id"] for r in graph.structuredContent["rows"]])))
                results.append(evidence.structuredContent)
                it = iter(results)
                journey(lambda r:next(it),config)
                for rec in bad_requests():
                    want = adapter.call(rec)
                    out = await session.call_tool(rec["op"], {k:v for k,v in rec.items() if k!="op"})
                    assert out.isError
                    assert out.structuredContent == want
                # Persistent connection repeats both result and warm graph path.
                again = await session.call_tool("describe",dict(corpus="demo"))
                assert clean(again.structuredContent) == clean(results[0])
    asyncio.run(run())


def bad_requests():
    return [request("bogus"), request("search",dict(text="deploy",vector=[0])),
            request("search",dict(text="deploy",vector=[0,0],candidates=4097)),
            request("fetch",dict(ids=[])), request("fetch",dict(ids=[True])),
            request("fetch",dict(ids=[1,1])), request("fetch",dict(ids=list(range(65)))),
            request("traverse",dict(ids=[1],depth=9)),
            request("search",dict(mode="lexical",text='"')),
            request("search",dict(vector=[0,0],text="deploy",tenant="other")),
            dict(op="describe",corpus="absent"), dict(op="fetch",corpus="demo",args=[],extra=1)]


@pytest.mark.parametrize("req",bad_requests())
def test_errors(adapter,req):
    assert not adapter.call(req)["ok"]


def test_scope_empty_and_truncation(adapter):
    fetched = adapter.call(request("fetch",dict(ids=[1,6,7,8,9,999])))
    assert [r["id"] for r in fetched["rows"]] == [1]
    assert fetched["missing_ids"] == [6,7,8,9,999]
    full = adapter.call(request("search",dict(text="deploy",vector=[0,0],candidates=10)))
    assert {r["id"] for r in full["rows"]} == {1,2,3,4,5,10}
    limited = adapter.call(request("search",dict(text="deploy",vector=[0,0],limit=1)))
    assert limited["truncated"] and limited["returned"] == 1
    empty = adapter.call(request("search",dict(mode="lexical",text="absentword")))
    assert empty["ok"] and not empty["rows"]
    single = adapter.call(request("search",dict(text="absentword",vector=[0,0])))
    assert single["ok"] and single["branches"]["keyword"]["returned"] == 0
    for r in single["rows"]:
        assert r["keyword_rank"] is None
    graph = adapter.call(request("traverse",dict(ids=[1,5,6,999],depth=0)))
    assert graph["rows"] == [dict(id=1,depth=0),dict(id=5,depth=0)]
    assert graph["missing_ids"] == [6,999]
    assert adapter.call(request("traverse",dict(ids=[1],depth=8,limit=1)))["truncated"]
    stored = adapter.call(request("search",dict(mode="dense",embedding_id=1)))
    supplied = adapter.call(request("search",dict(mode="dense",vector=[1,0])))
    assert stored["rows"] == supplied["rows"]
    assert adapter.call(request("search",dict(mode="dense",embedding_id=6)))["error"]["code"] == "missing_embedding"


def test_readonly_controls(adapter):
    for sql in ["delete from documents", "create temp table x(a)", "pragma query_only=off",
                "attach ':memory:' as other", "select load_extension('ignored')"]:
        with pytest.raises(sqlite3.DatabaseError):
            adapter.db.execute(sql)
    assert adapter.call(request("describe"))["ok"]


def test_uri_ro_without_authorizer(adapter):
    adapter.db.set_authorizer(None)
    adapter.db.execute("pragma query_only=off")
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        adapter.db.execute("delete from documents")


def test_cli_bad_records(adapter):
    source = io.BytesIO(b'not json\n'+b'x'*(LIMITS['input_bytes']+20)+b'\n'+json.dumps(request("describe")).encode()+b'\n')
    sink=io.BytesIO()
    assert cli(adapter,source,sink) == 1
    rows = [json.loads(r) for r in sink.getvalue().splitlines()]
    assert [r.get("error",{}).get("code") for r in rows] == ["invalid_json","input_limit",None]


def test_database_error(adapter):
    adapter.config["corpora"]["demo"]["documents"]="absent"
    assert adapter.call(request("describe"))["error"]["code"] == "database_error"


def test_output_cap(adapter,monkeypatch):
    monkeypatch.setitem(LIMITS,"output_bytes",500)
    assert adapter.call(request("fetch",dict(ids=[1,2,3,4,5])))["error"]["code"] == "output_limit"


def test_real_transport_caps(config):
    cfg=json.loads(config.read_text())
    with sqlite3.connect(cfg["database"]) as writer:
        writer.execute("update documents set body=? where id=1",("x"*270000,))
    records=[request("fetch",dict(ids=[1])),request("search",dict(text="x"*65536,vector=[0,0]))]
    p=subprocess.run([sys.executable,str(HERE/"adapter.py"),"--config",str(config),"cli"],
                     input="".join(json.dumps(r)+"\n" for r in records),text=True,capture_output=True)
    assert p.returncode==1
    assert [json.loads(r)["error"]["code"] for r in p.stdout.splitlines()]==["output_limit","input_limit"]
    async def run():
        params=StdioServerParameters(command=sys.executable,args=[str(HERE/"adapter.py"),"--config",str(config),"mcp"])
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize()
                for rec,code in zip(records,["output_limit","input_limit"]):
                    out=await session.call_tool(rec["op"],dict(corpus="demo",args=rec["args"]))
                    assert out.isError and out.structuredContent["error"]["code"]==code
    asyncio.run(run())


@pytest.mark.parametrize("vector",[[True,0],[1e11,0],["1",0],[],[0,0,0]])
def test_invalid_vectors(adapter,vector):
    out=adapter.call(request("search",dict(mode="dense",vector=vector)))
    assert out["error"]["code"]=="invalid_vector"


def test_admission_caps(adapter,monkeypatch):
    monkeypatch.setitem(LIMITS,"database_bytes",1)
    assert adapter.call(request("describe"))["error"]["code"]=="workload_limit"


def test_corpus_admission(config):
    cfg=json.loads(config.read_text())
    with sqlite3.connect(cfg["database"]) as writer:
        writer.executemany("insert into documents(id) values (?)",((n,) for n in range(100,10101)))
    a=Adapter(config)
    try:
        assert a.call(request("describe"))["error"]["code"]=="workload_limit"
    finally:
        a.close()


def test_graph_admission_and_snapshot(config):
    a=Adapter(config)
    try:
        assert a.call(request("traverse",dict(ids=[1])))["ok"]
        d=sqlite3.connect(json.loads(config.read_text())["database"])
        # A later call observes a later commit, including graph-cache invalidation.
        d.execute("insert into edges values (1,5)")
        d.commit()
        assert 5 in [r["id"] for r in a.call(request("traverse",dict(ids=[1])))["rows"]]
        a.config["corpora"]["demo"]["edges"]="edges"
        assert a.call(request("traverse",dict(ids=[1])))["error"]["code"] == "graph_scope"
        a.config["corpora"]["demo"]["edges"]="admitted_edges"
        d.executemany("insert into edges values (1,2)",[()] * 20001)
        d.commit()
        assert a.call(request("traverse",dict(ids=[1])))["error"]["code"] == "workload_limit"
        d.close()
    finally:
        a.close()


def test_graph_cache_reuse(adapter):
    trace=[]
    adapter.db.set_trace_callback(trace.append)
    query=request("traverse",dict(ids=[1],depth=2))
    assert adapter.call(query)["ok"]
    build='-- SELECT "src", "dst" FROM "admitted_edges"'
    assert trace.count(build)==1
    assert adapter.call(query)["ok"]
    assert trace.count(build)==1


def test_operation_snapshot(config):
    cfg=json.loads(config.read_text())
    writer=sqlite3.connect(cfg["database"])
    writer.execute("pragma journal_mode=wal")
    a=Adapter(config)
    original=a.fetch
    def fetch_during_commit(c,args):
        # call() has already read admission counts, establishing its snapshot.
        writer.execute("update documents set body='new evidence' where id=1")
        writer.commit()
        return original(c,args)
    a.fetch=fetch_during_commit
    try:
        rec=request("fetch",dict(ids=[1]))
        assert a.call(rec)["rows"][0]["body"]=="Deploy a small database safely."
        a.fetch=original
        assert a.call(rec)["rows"][0]["body"]=="new evidence"
    finally:
        a.close()
        writer.close()


def test_cutoff_ties_are_deterministic(adapter):
    got=adapter.call(request("search",dict(mode="dense",vector=[0,0],candidates=3)))
    assert [r["id"] for r in got["rows"]]==[1,2,3]
    assert [r["vector_rank"] for r in got["rows"]]==[1,2,3]


def test_input_cap_and_non_json(adapter):
    for value in [None, [], "text", {"op":[]}, {"op":"describe","corpus":[]},
                  request("search",dict(text="x"*65536,vector=[0,0]))]:
        assert not adapter.call(value)["ok"]


def test_empty_dense_and_all_branches(config):
    cfg=json.loads(config.read_text())
    cfg["corpora"]["demo"]["tenant"]="no-records"
    config.write_text(json.dumps(cfg))
    a=Adapter(config)
    try:
        for mode in ("dense","hybrid","lexical"):
            args=dict(mode=mode)
            if mode!="lexical": args["vector"]=[0,0]
            if mode!="dense": args["text"]="deploy"
            out=a.call(request("search",args))
            assert out["ok"] and not out["rows"]
    finally:
        a.close()


def test_startup_missing_database(tmp_path):
    p = tmp_path/"config.json"
    p.write_text(json.dumps(dict(database="absent.db",extension=str(EXTENSION),corpora={})))
    run=subprocess.run([sys.executable,str(HERE/"adapter.py"),"--config",str(p),"mcp"],capture_output=True,text=True)
    assert run.returncode==2 and not run.stdout
    assert json.loads(run.stderr)["error"]["code"]=="startup_error"
    assert not (tmp_path/"absent.db").exists()
