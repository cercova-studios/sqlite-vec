"""Bounded Lance setup probes; retain stalls instead of treating them as passes."""
from pathlib import Path
import json,os,subprocess,sys
HERE=Path(__file__).resolve().parent
CODE='''
import tempfile,lancedb,pyarrow as pa
from lancedb.index import BTree,HnswFlat
pa.set_cpu_count(1);pa.set_io_thread_count(1)
with tempfile.TemporaryDirectory() as root:
 print('connect',flush=True);db=lancedb.connect(root)
 table=pa.table({'id':list(range(512)),'bucket':list(range(512)),'vector':pa.FixedSizeListArray.from_arrays(pa.array([float(i%7) for i in range(512*8)],type=pa.float32()),8)})
 print('create table',flush=True);t=db.create_table('docs',table)
 print('scalar index',flush=True);t.create_index('bucket',config=BTree())
 print('vector index',flush=True);t.create_index('vector',config=HnswFlat(num_partitions=2,m=16,ef_construction=200))
 print('query',flush=True);print(t.search([0.]*8).select(['id','_distance']).limit(10).to_list(),flush=True)
 print('complete',flush=True)
'''
results=[]
for cpu,io in [(1,1),(1,2),(2,1),(2,2)]:
 env=os.environ.copy();env.update(LANCE_CPU_THREADS=str(cpu),LANCE_IO_THREADS=str(io),RAYON_NUM_THREADS='1')
 try:
  p=subprocess.run([sys.executable,'-c',CODE],env=env,capture_output=True,text=True,timeout=15)
  result={'cpu':cpu,'io':io,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 except subprocess.TimeoutExpired as e:
  result={'cpu':cpu,'io':io,'timeout_seconds':15,'stdout':(e.stdout or b'').decode(),'stderr':(e.stderr or b'').decode()}
 results.append(result);print(cpu,io,'timeout' if 'timeout_seconds' in result else result['returncode'],flush=True)
(HERE/'recorded/thread-probes.json').write_text(json.dumps(results,indent=2)+'\n')
