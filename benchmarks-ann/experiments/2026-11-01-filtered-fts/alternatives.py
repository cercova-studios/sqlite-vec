"""Compare eligibility SQL and planner-statistics controls; require identical rows."""
from pathlib import Path
import sys,importlib.util,json,time
import numpy as np
root=Path(__file__).resolve().parents[3];old=root/'benchmarks-ann/experiments/2026-10-26-scifact-hybrid';work=root/'benchmarks-ann/experiments/2026-11-01-filtered-fts/.work'
spec=importlib.util.spec_from_file_location('scifact',old/'evaluate.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
data=json.loads((old/'.work/data.json').read_text());db=s.connect(':memory:',work/'build/vec0.dylib');s.setup(db,data['documents'],np.load(old/'.work/documents.npy'))
q=data['queries'][1];p=dict(tenant='ours',as_of=100,q_text=s.lexical_queries([q])[q['_id']],branch_limit=50)
exists=s.KEYWORD.replace('rowid in (select id from eligible)','exists (select 1 from eligible where eligible.id = fts_docs.rowid)')
r=[];expected=None
for name,sql in [('baseline',s.KEYWORD),('exists',exists),('optimize',s.KEYWORD),('analyze',s.KEYWORD)]:
 if name=='optimize':db.execute('pragma optimize')
 if name=='analyze':db.execute('analyze')
 sql='/* '+name+' */ '+sql
 t=time.perf_counter();rows=db.execute(sql,p).fetchall();seconds=time.perf_counter()-t
 if expected is None:expected=rows
 assert expected==rows
 plan=db.execute('explain query plan '+sql,p).fetchall();print(name,seconds,plan,flush=True)
 r.append(dict(variant=name,seconds=seconds,plan=plan,rows=rows))
(work/'alternatives.json').write_text(json.dumps(r,indent=2)+'\n');db.close()
