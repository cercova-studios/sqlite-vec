import asyncio
import json
from pathlib import Path
import sqlite3
import subprocess

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rust_test_client import RustAdapter, command
from fixture import create
from graph_test_data import colliding_ids


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
    a = RustAdapter(config)
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
    # Rust verifies this field against its own bundled SQLite in its unit suite.
    assert tuple(map(int,description["sqlite_version"].split("."))) >= (3,39,0)
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
    p = subprocess.run(command(config),
                       input="".join(json.dumps(r)+"\n" for r in records),text=True,capture_output=True,check=True)
    results = iter(json.loads(line) for line in p.stdout.splitlines())
    journey(lambda r:next(results),config)
    assert not list(results)
    assert not p.stderr


def test_persistent_cli_journey(adapter,config):
    journey(adapter.call, config)


def test_mcp_journey_and_errors(config,adapter):
    async def run():
        cmd=command(config,"mcp")
        params = StdioServerParameters(command=cmd[0],args=cmd[1:])
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert [t.name for t in tools.tools] == ["describe","search","traverse","fetch","reach","witness"]
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


def test_cli_bad_records(config):
    source=b'not json\n'+b'x'*(65536+20)+b'\n'+json.dumps(request("describe")).encode()+b'\n'
    run=subprocess.run(command(config),input=source,capture_output=True)
    assert run.returncode==1 and not run.stderr
    rows = [json.loads(r) for r in run.stdout.splitlines()]
    assert [r.get("error",{}).get("code") for r in rows] == ["invalid_json","input_limit",None]


def test_database_error(adapter,config):
    assert adapter.call(request("describe"))["ok"]
    with sqlite3.connect(json.loads(config.read_text())["database"]) as writer:
        writer.execute("drop table documents")
    assert adapter.call(request("describe"))["error"]["code"] == "database_error"


def test_real_transport_caps(config):
    cfg=json.loads(config.read_text())
    with sqlite3.connect(cfg["database"]) as writer:
        writer.execute("update documents set body=? where id=1",("x"*270000,))
    records=[request("fetch",dict(ids=[1])),request("search",dict(text="x"*65536,vector=[0,0]))]
    p=subprocess.run(command(config),
                     input="".join(json.dumps(r)+"\n" for r in records),text=True,capture_output=True)
    assert p.returncode==1
    assert [json.loads(r)["error"]["code"] for r in p.stdout.splitlines()]==["output_limit","input_limit"]
    async def run():
        cmd=command(config,"mcp")
        params=StdioServerParameters(command=cmd[0],args=cmd[1:])
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


def test_corpus_admission(config):
    cfg=json.loads(config.read_text())
    with sqlite3.connect(cfg["database"]) as writer:
        writer.executemany("insert into documents(id) values (?)",((n,) for n in range(100,10101)))
    a=RustAdapter(config)
    try:
        assert a.call(request("describe"))["error"]["code"]=="workload_limit"
    finally:
        a.close()


def test_graph_admission_and_snapshot(config):
    a=RustAdapter(config)
    try:
        assert a.call(request("traverse",dict(ids=[1])))["ok"]
        d=sqlite3.connect(json.loads(config.read_text())["database"])
        # A later call observes a later commit, including graph-cache invalidation.
        d.execute("insert into edges values (1,5)")
        d.commit()
        assert 5 in [r["id"] for r in a.call(request("traverse",dict(ids=[1])))["rows"]]
        a.close()
        cfg=json.loads(config.read_text());cfg["corpora"]["demo"]["edges"]="edges"
        config.write_text(json.dumps(cfg));a=RustAdapter(config)
        assert a.call(request("traverse",dict(ids=[1])))["error"]["code"] == "graph_scope"
        a.close()
        cfg["corpora"]["demo"]["edges"]="admitted_edges"
        config.write_text(json.dumps(cfg));a=RustAdapter(config)
        d.executemany("insert into edges values (1,2)",[()] * 20001)
        d.commit()
        assert a.call(request("traverse",dict(ids=[1])))["error"]["code"] == "workload_limit"
        d.close()
    finally:
        a.close()


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
    a=RustAdapter(config)
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
    run=subprocess.run(command(p,"mcp"),capture_output=True,text=True)
    assert run.returncode==2 and not run.stdout
    assert json.loads(run.stderr)["error"]["code"]=="startup_error"
    assert not (tmp_path/"absent.db").exists()


def test_cli_flag_order_and_help(config):
    binary=command(config)[0]
    for args in [["cli","--config",str(config)],[f"--config={config}","cli"]]:
        run=subprocess.run([binary,*args],input=json.dumps(request("describe")),text=True,capture_output=True)
        assert run.returncode==0 and json.loads(run.stdout)["ok"] and not run.stderr
    help_run=subprocess.run([binary,"--help"],text=True,capture_output=True)
    assert help_run.returncode==0 and "--config" in help_run.stdout and not help_run.stderr


def test_rust_runtime_needs_no_python(config):
    run=subprocess.run(command(config),input=json.dumps(request("describe")),text=True,capture_output=True,
                       env={"PATH":"/nonexistent"})
    assert run.returncode==0 and json.loads(run.stdout)["ok"] and not run.stderr


@pytest.fixture
def witness_config(config):
    cfg = json.loads(config.read_text())
    with sqlite3.connect(cfg['database']) as db:
        db.execute('create table evidence_edges(id integer primary key,src,dst)')
        db.executemany('insert into evidence_edges values (?,?,?)', [(101,1,2),(102,2,4),(103,1,3),(104,3,4),(105,1,2)])
    cfg['corpora']['demo']['edges'] = 'evidence_edges'
    cfg['corpora']['demo']['graph']['edge_id'] = 'id'
    config.write_text(json.dumps(cfg))
    return config


def witness_request(**updates):
    args = dict(version='g2/1',node_type='Function',edge_type='CALLS',direction='out',source=1,target=4,work=100000,bytes=65536)
    args.update(updates)
    return request('witness',args)


def test_witness_cli_failures_and_recovery(witness_config):
    adapter = RustAdapter(witness_config)
    try:
        result = adapter.call(witness_request())
        assert result['ok'] and result['path'] in ([[1,None],[2,101],[4,102]], [[1,None],[3,103],[4,104]])
        assert adapter.call(witness_request(work=1))['error']['code'] == 'work_limit'
        assert adapter.call(witness_request(bytes=128))['error']['code'] == 'memory_limit'
        assert adapter.call(witness_request())['ok']
        assert adapter.call(witness_request(source=6))['missing_ids'] == [6]
        assert adapter.call(witness_request(target=1))['path'] == [[1,None]]
        assert adapter.call(request('fetch', {'ids':[1]}))['ok']
    finally:
        adapter.close()


def test_witness_mcp_parity(witness_config):
    async def run():
        cmd = command(witness_config, 'mcp')
        async with stdio_client(StdioServerParameters(command=cmd[0],args=cmd[1:])) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize()
                record = witness_request()
                result = await session.call_tool('witness', {k:v for k,v in record.items() if k != 'op'})
                assert result.structuredContent['ok']
                assert result.structuredContent['path'][0] == [1,None]
    asyncio.run(run())



@pytest.mark.parametrize("cancel_queued", [False, True])
def test_witness_mcp_cancel_then_reuse(witness_config, cancel_queued):
    cfg = json.loads(witness_config.read_text())
    with sqlite3.connect(cfg['database']) as db:
        db.execute('delete from evidence_edges')
        # Collision-heavy validation keeps native work pending for cancellation.
        db.executemany('insert into evidence_edges values (?,?,?)', [(i,1,2) for i in colliding_ids(19999)])
    async def run():
        cmd = command(witness_config, 'mcp')
        async with stdio_client(StdioServerParameters(command=cmd[0],args=cmd[1:])) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize()
                record = witness_request(work=1000000000,bytes=4*1024*1024)
                call = asyncio.create_task(session.call_tool('witness', {k:v for k,v in record.items() if k != 'op'}))
                await asyncio.sleep(0.01)
                assert not call.done(), 'cancellation workload must still be running'
                if cancel_queued:
                    queued = asyncio.create_task(session.call_tool('fetch', {'corpus':'demo','args':{'ids':[1]}}))
                    await asyncio.sleep(0.01)
                    queued.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await queued
                    completed = await asyncio.wait_for(call, timeout=5)
                    assert completed.structuredContent['ok'], completed
                else:
                    call.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await call
                # Discovery must stay responsive while a native call is stopping.
                tools = await asyncio.wait_for(session.list_tools(), timeout=5)
                assert any(t.name == 'witness' for t in tools.tools)
                response = await asyncio.wait_for(session.call_tool('fetch', {'corpus':'demo','args':{'ids':[1]}}), timeout=5)
                assert response.structuredContent['ok']
                assert response.structuredContent['rows'][0]['id'] == 1
    asyncio.run(run())


@pytest.mark.parametrize('mutation,code', [
    ('missing_column','schema_error'),('duplicate_id','database_error'),
    ('null_id','database_error'),('wrong_type','unsupported')])
def test_witness_mapping_rejection(witness_config, mutation, code):
    cfg = json.loads(witness_config.read_text())
    if mutation == 'missing_column':
        cfg['corpora']['demo']['graph']['edge_id'] = 'missing'
    elif mutation in ('duplicate_id','null_id'):
        with sqlite3.connect(cfg['database']) as db:
            if mutation == 'duplicate_id':
                db.execute('create view bad_ids as select * from evidence_edges union all select * from evidence_edges')
            else:
                db.execute('create view bad_ids as select NULL as id,src,dst from evidence_edges')
        cfg['corpora']['demo']['edges'] = 'bad_ids'
    witness_config.write_text(json.dumps(cfg))
    adapter = RustAdapter(witness_config)
    try:
        query = witness_request(node_type='File') if mutation == 'wrong_type' else witness_request()
        assert adapter.call(query)['error']['code'] == code
        assert adapter.call(request('fetch', {'ids':[1]}))['ok']
    finally:
        adapter.close()
