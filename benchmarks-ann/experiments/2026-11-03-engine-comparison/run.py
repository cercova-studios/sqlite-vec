"""Run isolated workers sequentially with recorded resource settings."""
from pathlib import Path
import hashlib,json,os,platform,random,subprocess,sys
HERE=Path(__file__).resolve().parent
ENGINES=['upstream-flat','fork-flat','fork-hnsw','zvec-hnsw','lance-exact','lance-ann','lance-hnsw1']
def main():
    env=os.environ.copy()
    limits={key:'1' for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','RAYON_NUM_THREADS','LANCE_CPU_THREADS','LANCE_IO_THREADS']}
    limits['LANCE_IO_THREADS']='2'
    env.update(limits)
    order=ENGINES.copy();random.Random(23).shuffle(order)
    provenance={'platform':platform.platform(),'python':sys.version,'limits':limits,'engine_order':order,'packages':{}}
    import importlib.metadata
    provenance['packages']={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    provenance['worker_sha256']=hashlib.sha256((HERE/'worker.py').read_bytes()).hexdigest()
    (HERE/'recorded/runtime.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for dataset in ['scifact','sift50k']:
        for engine in order:
            print(dataset,engine,flush=True)
            with (HERE/'recorded'/f'{dataset}-{engine}.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'worker.py'),engine,dataset],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
if __name__=='__main__':main()
