"""Separate SQLite IN-set construction from the native graph callback."""
from pathlib import Path
import json,sqlite3,statistics,time
import numpy as np
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'2026-11-03-engine-comparison'
db=sqlite3.connect(HERE/'.work/filter.db',isolation_level=None)
db.enable_load_extension(True);db.load_extension(str(PREVIOUS/'.work/build/vec0.dylib'));db.enable_load_extension(False)
queries=np.load(PREVIOUS/'.work/sift50k/queries.npy')[:20]
controls={
 'count':('select count(*) from scope where bucket<100',lambda q:[]),
 'materialize_in':('select ? in (select id from scope where bucket<100)',lambda q:[-1]),
 'ordered_in':('select ? in (select id from scope where bucket<100 order by id)',lambda q:[-1]),
 'knn':('select rowid,distance from t where v match ? and k=10 and rowid in (select id from scope where bucket<100) order by distance',lambda q:[q.tobytes()]),
 'ordered_knn':('select rowid,distance from t where v match ? and k=10 and rowid in (select id from scope where bucket<100 order by id) order by distance',lambda q:[q.tobytes()]),
}
raw=[];plans={}
for name,(sql,args) in controls.items():
 db.execute(sql,args(queries[0])).fetchall()
 plans[name]=[list(r) for r in db.execute('explain query plan '+sql,args(queries[0]))]
 for repeat in range(3):
  for i,q in enumerate(queries):
   start=time.perf_counter_ns();rows=db.execute(sql,args(q)).fetchall();elapsed=time.perf_counter_ns()-start
   if name=='ordered_knn':assert rows==db.execute(controls['knn'][0],[q.tobytes()]).fetchall()
   raw.append({'name':name,'round':repeat,'query':i,'ms':elapsed/1e6})
summary={name:statistics.median(r['ms'] for r in raw if r['name']==name) for name in controls}
(HERE/'recorded/sql-controls.json').write_text(json.dumps({'summary_ms':summary,'measurements':raw,'plans':plans},indent=2)+'\n')
print(json.dumps(summary,indent=2))
