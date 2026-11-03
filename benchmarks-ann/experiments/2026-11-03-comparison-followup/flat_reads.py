"""Count vector BLOB bytes for a sparse flat query; not a timing profile."""
from pathlib import Path
import json,shutil,sqlite3,subprocess,sys
import numpy as np
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'2026-11-03-engine-comparison'
REPO=HERE.parents[2]
def child():
    data=PREVIOUS/'.work/sift50k';x=np.load(data/'vectors.npy');bucket=np.load(data/'bucket.npy');order=np.load(data/'order.npy');q=np.load(data/'queries.npy')[0]
    db=sqlite3.connect(':memory:',isolation_level=None);db.enable_load_extension(True);db.load_extension(str(HERE/'.work/flat/vec0.dylib'));db.enable_load_extension(False)
    db.execute('create virtual table t using vec0(v float[128])');db.execute('create table scope(id integer primary key,bucket integer)');db.execute('create index sb on scope(bucket)')
    db.execute('begin');db.executemany('insert into t(rowid,v) values (?,?)',((int(i),x[i].tobytes()) for i in order));db.executemany('insert into scope values (?,?)',((int(i),int(bucket[i])) for i in order));db.execute('commit')
    rows=db.execute('select rowid,distance from t where v match ? and k=10 and rowid in (select id from scope where bucket<10) order by distance',[q.tobytes()]).fetchall()
    eligible=np.flatnonzero(bucket<10);d2=np.sum((x[eligible].astype('float64')-q.astype('float64'))**2,axis=1);cutoff=np.sort(d2)[9]
    assert all(i in eligible for i,_ in rows) and len(rows)==10
    actual=np.sum((x[[r[0] for r in rows]].astype('float64')-q.astype('float64'))**2,axis=1)
    assert np.count_nonzero(actual<cutoff)==np.count_nonzero(d2<cutoff) and np.all(actual<=cutoff)
    print(json.dumps({'rows':rows,'eligible_count':len(eligible),'eligible_vector_bytes':len(eligible)*x.shape[1]*4}))
if __name__=='__main__':
    if '--child' in sys.argv:child();sys.exit(0)
    target=HERE/'.work/flat';shutil.copytree(PREVIOUS/'.work/build',target,dirs_exist_ok=True)
    p=target/'sqlite-vec.c';s=p.read_text();anchor='rc = sqlite3_blob_read(blobVectors, baseVectors, currentBaseVectorsSize, 0);';assert s.count(anchor)==1;s=s.replace(anchor,anchor+'\n    fprintf(stderr,"BLOB_BYTES %lld\\n",currentBaseVectorsSize);');p.write_text(s)
    subprocess.run(['cc','-O3','-fPIC','-shared','-mcpu=apple-m1','-DSQLITE_VEC_ENABLE_NEON','-I'+str(REPO/'vendor'),str(p),'-o',str(target/'vec0.dylib')],check=True)
    result=subprocess.run([sys.executable,__file__,'--child'],capture_output=True,text=True,check=True)
    values=[int(line.split()[1]) for line in result.stderr.splitlines() if line.startswith('BLOB_BYTES ')]
    raw=json.loads(result.stdout);raw['read_sizes']=values;raw['vector_blob_reads']=len(values);raw['vector_blob_bytes']=sum(values);raw['read_amplification']=sum(values)/raw['eligible_vector_bytes']
    (HERE/'recorded/flat-reads.json').write_text(json.dumps(raw,indent=2)+'\n');print(json.dumps(raw))
