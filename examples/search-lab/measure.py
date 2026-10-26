"""Measure startup and warm calls on the fixture; writes raw JSON to stdout."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

from adapter import Adapter


HERE = Path(__file__).resolve().parent
RECORDS = [dict(op="describe",corpus="demo"),
           dict(op="search",corpus="demo",args=dict(text="deploy",vector=[0,0],candidates=3,limit=3)),
           dict(op="traverse",corpus="demo",args=dict(ids=[1,2,3],depth=2)),
           dict(op="fetch",corpus="demo",args=dict(ids=[1,2,3,4]))]


def checked(result):
    assert result["ok"], result
    return result


def summary(raw):
    assert len(raw)>=20 and all(x>=0 for x in raw)
    return dict(samples_ms=raw,p50_ms=statistics.median(raw),p95_ms=sorted(raw)[int(.95*len(raw))-1])


def measure(config):
    out={}
    t=time.perf_counter()
    a=Adapter(config)
    out["handler_connect_ms"]=(time.perf_counter()-t)*1000
    out["sqlite_version"]=a.db.execute("select sqlite_version()").fetchone()[0]
    out["extension_version"]=a.extension_version
    out["handler"]={}
    traces=[]
    a.db.set_trace_callback(traces.append)
    for rec in RECORDS:
        t=time.perf_counter()
        checked(a.call(rec))
        first=(time.perf_counter()-t)*1000
        times=[]
        for _ in range(20):
            t=time.perf_counter(); checked(a.call(rec)); times.append((time.perf_counter()-t)*1000)
        out["handler"][rec["op"]]=dict(first_ms=first,**summary(times))
    out["observed_native_graph_builds"]=traces.count('-- SELECT "src", "dst" FROM "admitted_edges"')
    assert out["observed_native_graph_builds"]==1
    a.close()
    out["cli"]={}
    t=time.perf_counter()
    p=subprocess.Popen([sys.executable,str(HERE/"adapter.py"),"--config",str(config),"cli"],
                       stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for rec in RECORDS:
            def call():
                p.stdin.write(json.dumps(rec).encode()+b"\n");p.stdin.flush()
                return checked(json.loads(p.stdout.readline()))
            call()
            first=(time.perf_counter()-t)*1000
            times=[]
            for _ in range(20):
                t=time.perf_counter();call();times.append((time.perf_counter()-t)*1000)
            out["cli"][rec["op"]]=dict(first_ms=first,**summary(times))
            t=time.perf_counter()
        p.stdin.close()
        assert p.wait(timeout=10)==0
        assert not p.stderr.read()
    finally:
        if p.poll() is None:
            p.kill();p.wait()
        p.stdout.close();p.stderr.close()
    out["mcp"]=asyncio.run(measure_mcp(config))
    out["rss"]={"unit":"bytes" if sys.platform=="darwin" else "KiB",
                "parent_process_high_water":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "largest_terminated_child_high_water":resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                "scope":"parent includes MCP client; child maximum across CLI and MCP, not combined RSS"}
    cfg=json.loads(Path(config).read_text())
    out["environment"]=dict(platform=platform.platform(),machine=platform.machine(),python=sys.version,
        mcp=importlib.metadata.version("mcp"),fixture_sha256=hashlib.sha256(Path(cfg["database"]).read_bytes()).hexdigest(),
        extension_sha256=hashlib.sha256(Path(cfg["extension"]).read_bytes()).hexdigest(),
        workload="10 original-text rows, 6 eligible, 7 admitted edges, 2-dimensional L2; warm OS cache",
        limits="first process/connection calls are not cold-disk measurements; one run, 20 warm calls per operation")
    out["source_sha256"]={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                          for name in ("adapter.py","fixture.py","measure.py")}
    return out


async def measure_mcp(config):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    out={}
    t=time.perf_counter()
    params=StdioServerParameters(command=sys.executable,args=[str(HERE/"adapter.py"),"--config",str(config),"mcp"])
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize()
            out["spawn_and_initialize_ms"]=(time.perf_counter()-t)*1000
            for rec in RECORDS:
                async def call():
                    result=await session.call_tool(rec["op"],dict(corpus=rec["corpus"],args=rec.get("args",{})))
                    assert not result.isError, result
                    return checked(result.structuredContent)
                t=time.perf_counter();await call();first=(time.perf_counter()-t)*1000
                times=[]
                for _ in range(20):
                    t=time.perf_counter();await call();times.append((time.perf_counter()-t)*1000)
                out[rec["op"]]=dict(first_ms=first,**summary(times))
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True,type=Path)
    print(json.dumps(measure(p.parse_args().config),indent=2))
