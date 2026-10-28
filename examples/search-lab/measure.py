"""Measure the Rust CLI/MCP transports on the fixture; Python is only the client."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from rust_test_client import BINARY,HERE,command

RECORDS=[dict(op="describe",corpus="demo"),
         dict(op="search",corpus="demo",args=dict(text="deploy",vector=[0,0],candidates=3,limit=3)),
         dict(op="traverse",corpus="demo",args=dict(ids=[1,2,3],depth=2)),
         dict(op="fetch",corpus="demo",args=dict(ids=[1,2,3,4]))]


def summary(first,samples):
    assert len(samples)==20 and all(t>=0 for t in samples)
    return dict(first_ms=first,samples_ms=samples,p50_ms=statistics.median(samples),p95_ms=sorted(samples)[18])


def measure(config):
    result={"cli":{}}
    t=time.perf_counter()
    proc=subprocess.Popen(command(config),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for record in RECORDS:
            def call():
                proc.stdin.write(json.dumps(record).encode()+b"\n");proc.stdin.flush()
                value=json.loads(proc.stdout.readline())
                assert value["ok"],value
                return value
            value=call();first=(time.perf_counter()-t)*1000
            if record["op"]=="describe":
                result["sqlite_version"]=value["sqlite_version"]
                result["extension_version"]=value["extension_version"]
            samples=[]
            for _ in range(20):
                t=time.perf_counter();call();samples.append((time.perf_counter()-t)*1000)
            result["cli"][record["op"]]=summary(first,samples)
            t=time.perf_counter()
        proc.stdin.close();assert proc.wait(timeout=10)==0
        assert not proc.stderr.read()
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        proc.stdout.close();proc.stderr.close()
    result["mcp"]=asyncio.run(mcp_measure(config))
    result["child_peak_rss"]={"value":resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "unit":"bytes" if sys.platform=="darwin" else "KiB", "scope":"largest terminated CLI/MCP child, not combined RSS or client RSS"}
    cfg=json.loads(config.read_text())
    result["environment"]=dict(platform=platform.platform(),client_python=sys.version,
        workload="10 documents, 6 eligible, 2-dimensional L2, 7 admitted edges; warm OS cache; 20 sequential warm samples per operation",
        limits="first CLI describe includes spawn; MCP startup is separate; no cold-disk or production claims")
    files={"fixture":Path(cfg["database"]),"extension":Path(cfg["extension"]),"binary":BINARY,
           **{p:HERE/p for p in ["src/lib.rs","src/main.rs","src/tool-arguments.json","Cargo.lock"]}}
    result["sha256"]={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    return result


async def mcp_measure(config):
    result={};cmd=command(config,"mcp");t=time.perf_counter()
    async with stdio_client(StdioServerParameters(command=cmd[0],args=cmd[1:])) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize()
            result["spawn_and_initialize_ms"]=(time.perf_counter()-t)*1000
            for record in RECORDS:
                async def call():
                    value=await session.call_tool(record["op"],dict(corpus="demo",args=record.get("args",{})))
                    assert not value.isError and value.structuredContent["ok"],value
                t=time.perf_counter();await call();first=(time.perf_counter()-t)*1000
                samples=[]
                for _ in range(20):
                    t=time.perf_counter();await call();samples.append((time.perf_counter()-t)*1000)
                result[record["op"]]=summary(first,samples)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    print(json.dumps(measure(parser.parse_args().config),indent=2))
