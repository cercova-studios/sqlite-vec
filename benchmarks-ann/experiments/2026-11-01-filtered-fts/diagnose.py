from pathlib import Path
import sys,importlib.util,json,time,collections,os,subprocess
import numpy as np
repo=Path(__file__).resolve().parents[3];parent=repo/'benchmarks-ann/experiments/2026-10-26-scifact-hybrid'
spec=importlib.util.spec_from_file_location('scifact',parent/'evaluate.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
work=repo/'benchmarks-ann/experiments/2026-11-01-filtered-fts/.work'
data=json.loads((parent/'.work/data.json').read_text());db=s.connect(':memory:',work/'build/vec0.dylib');s.setup(db,data['documents'],np.load(parent/'.work/documents.npy'))
words=s.lexical_queries(data['queries']);query=data['queries'][1];p=dict(tenant='ours',as_of=100,q_text=words[query['_id']],branch_limit=50)
if '--worker' in sys.argv:
 (work/'ready.json').write_text(json.dumps({'pid':os.getpid()}));end=time.monotonic()+8
 while time.monotonic()<end:db.execute(s.KEYWORD,p).fetchall()
 db.close();sys.exit()
report={'query_id':query['_id'],'variants':{}}
for name,sql in [('baseline',s.KEYWORD),('unary',s.KEYWORD.replace('and rowid in','and +rowid in')),('exists',s.KEYWORD.replace('rowid in (select id from eligible)','exists (select 1 from eligible where eligible.id = fts_docs.rowid)'))]:
 plan=db.execute('explain query plan '+sql,p).fetchall();opcodes=db.execute('explain '+sql,p).fetchall()
 t=time.perf_counter();c=time.process_time();rows=db.execute(sql,p).fetchall();wall=time.perf_counter()-t;cpu=time.process_time()-c
 count=[0];trace=collections.Counter()
 def progress():count[0]+=1;return 0
 db.set_progress_handler(progress,10000);db.set_trace_callback(lambda q:trace.update([q]))
 t=time.perf_counter();instrumented=db.execute(sql,p).fetchall();profile_wall=time.perf_counter()-t
 db.set_progress_handler(None,0);db.set_trace_callback(None);assert rows==instrumented
 if name=='baseline':expected=rows
 else:assert rows==expected
 report['variants'][name]=dict(plan=plan,opcodes=opcodes,wall_seconds=wall,cpu_seconds=cpu,instrumented_seconds=profile_wall,vm_steps_lower_bound=count[0]*10000,trace_calls=sum(trace.values()),top_traces=trace.most_common(12),rows=rows)
 print(name,wall,cpu,count[0]*10000,sum(trace.values()),plan,flush=True)
(work/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');db.close()
(work/'ready.json').unlink(missing_ok=True)
log=(work/'worker.log').open('w');child=subprocess.Popen([sys.executable,__file__,'--worker'],stdout=log,stderr=log)
try:
 deadline=time.monotonic()+10
 while not (work/'ready.json').exists():
  assert child.poll() is None
  assert time.monotonic()<deadline
  time.sleep(.02)
 result=subprocess.run(['/usr/bin/sample',str(child.pid),'2','1','-file',str(work/'native-sample.txt')],capture_output=True,text=True,timeout=10)
 (work/'sample-status.json').write_text(json.dumps(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
 print('sample',result.returncode,result.stderr[:400],flush=True)
 child.wait(timeout=15)
finally:
 if child.poll() is None:child.terminate();child.wait()
 log.close()
