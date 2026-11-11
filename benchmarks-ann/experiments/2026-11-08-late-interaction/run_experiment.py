"""Explicitly run one frozen benchmark phase and record fresh-child resource use."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from freeze import HERE, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["score","live"])
    args=parser.parse_args()
    if sys.platform!="darwin":parser.error("resource units qualified for macOS only")
    work=HERE/".work"
    output=work/(args.phase+".json.gz")
    log=work/(args.phase+".log")
    if output.exists() or log.exists():parser.error("fresh output/log paths required; retain prior attempts explicitly")
    baseline=json.loads((work/"baseline-manifest.json").read_text())
    assert digest(work/"corpus.db")==baseline["combined_db_sha256"]
    embedding=json.loads((work/"embedding-manifest.json").read_text())
    for name,hashed in embedding["script_hashes"].items():assert digest(HERE/name)==hashed,name
    for name,hashed in embedding["files"].items():assert digest(work/name)==hashed,name
    model=json.loads((work/"model/manifest.json").read_text())
    for name,record in model["files"].items():assert digest(work/"model"/name)==record["sha256"],name
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",
             TOKENIZERS_PARALLELISM="false",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONHASHSEED="0")
    wrapper="""import json,resource,subprocess,sys,time
start=time.perf_counter()
with open(sys.argv[1],'x') as log:
 result=subprocess.run(sys.argv[2:],stdout=log,stderr=subprocess.STDOUT)
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
print(json.dumps(dict(exit_code=result.returncode,seconds=time.perf_counter()-start,peak_rss_bytes=usage.ru_maxrss,user_seconds=usage.ru_utime,system_seconds=usage.ru_stime)))
"""
    command=[sys.executable,"-c",wrapper,str(log),sys.executable,"-u",str(HERE/"benchmark.py"),args.phase,"--output",str(output)]
    result=subprocess.run(command,env=env,text=True,capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr)
    resource=json.loads(result.stdout)
    resource.update(phase=args.phase,log_sha256=digest(log),resource_units="macOS ru_maxrss bytes; one fresh child including imports and serialization")
    (work/(args.phase+"-resources.json")).write_text(json.dumps(resource,indent=2)+'\n')
    print(log.read_text())
    print(json.dumps(resource),flush=True)
    if resource["exit_code"]:raise SystemExit(resource["exit_code"])


if __name__=="__main__":main()
