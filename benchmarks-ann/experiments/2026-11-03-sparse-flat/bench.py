"""Compare whole-chunk, selective and adaptive reads on fixed public vectors."""
from pathlib import Path
import argparse, gzip, hashlib, json, platform, shutil, sqlite3, statistics, subprocess, sys, time
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PREVIOUS = HERE.parent / '2026-11-03-engine-comparison'
WORK = HERE / '.work'
BASE = 'dc20c85'
CASES = [('all','bucket',1000),('random50','bucket',500),('random10','bucket',100),
         ('random5','bucket',50),('random1','bucket',10),('random01','bucket',1),('band1','band',10),('empty','bucket',0)]
POLICY = '''  // Many small BLOB calls can cost more than copying the whole chunk. Keep
  // dense chunks contiguous, and charge 64 KiB of saved copying per read run.
  int kept = 0, runs = 0, previous = 0;
  for (int i = 0; i < count; i++) {
    int current = bitmap_get(selected, i);
    kept += current;
    runs += current && !previous;
    if (kept > count / 8 ||
        (i64)kept * stride + (i64)runs * 65536 >= (i64)count * stride)
      return sqlite3_blob_read(blob, data, count * stride, 0);
    previous = current;
  }
'''

def sha(data):
    return hashlib.sha256(data).hexdigest()

def connect(dataset, variant):
    db = sqlite3.connect(WORK/(dataset+'.db'),isolation_level=None)
    db.enable_load_extension(True)
    db.load_extension(str(WORK/variant/'vec0.dylib'))
    db.enable_load_extension(False)
    return db

def load(dataset):
    folder = PREVIOUS/'.work'/dataset
    x = np.load(folder/'vectors.npy',mmap_mode='r')
    return x,np.load(folder/'queries.npy'),np.load(folder/'bucket.npy'),np.load(folder/'band.npy'),np.load(folder/'order.npy')

def inputs(dataset):
    x,q,bucket,band,order = load('sift50k' if dataset == 'tiny2' else dataset)
    if dataset == 'tiny2':
        x = np.ascontiguousarray(x[:,:2]); q = np.ascontiguousarray(q[:,:2])
    return x,q,bucket,band,order

def worker(dataset,variant,count):
    x,q,bucket,band,_ = inputs(dataset)
    db = connect(dataset,variant)
    points, truth = [], {}
    for name,field,threshold in CASES:
        predicate = '' if name == 'all' else f' and rowid in (select id from scope where {field}<{threshold})'
        sql = 'select rowid,distance from t where v match ? and k=10'+predicate+' order by distance'
        allowed = np.flatnonzero((bucket if field == 'bucket' else band)<threshold)
        truth[name] = []
        for query in q[:min(count,5)]:
            db.execute(sql,[query.tobytes()]).fetchall()
        for i in range(count):
            start = time.perf_counter_ns()
            rows = db.execute(sql,[q[i].tobytes()]).fetchall()
            elapsed = (time.perf_counter_ns()-start)/1e6
            ids = [r[0] for r in rows]
            assert len(ids) == len(set(ids)) == min(10,len(allowed))
            assert all(j in allowed for j in ids)
            d2 = np.sum((x[allowed].astype('float64')-q[i].astype('float64'))**2,axis=1)
            returned = np.sum((x[ids].astype('float64')-q[i].astype('float64'))**2,axis=1)
            assert np.allclose([r[1] for r in rows],np.sqrt(returned),rtol=2e-5,atol=2e-5)
            if len(ids):
                cutoff = np.sort(d2)[len(ids)-1]
                assert np.count_nonzero(returned<cutoff) == np.count_nonzero(d2<cutoff) and np.all(returned<=cutoff)
                closer,ties = allowed[d2<cutoff].tolist(),allowed[d2==cutoff].tolist()
            else:
                closer,ties = [],[]
            truth[name].append(dict(k=len(ids),closer=closer,ties=ties))
            points.append(dict(case=name,query=i,ms=elapsed,rows=rows))
    db.close()
    print(json.dumps(dict(dataset=dataset,variant=variant,points=points,truth=truth)))

def build(variants):
    names = subprocess.check_output(['git','ls-tree','--name-only',BASE],cwd=REPO,text=True).splitlines()
    names = [n for n in names if n.startswith('sqlite-vec') and n.endswith('.c')]
    manifests = {}
    for variant in variants:
        folder = WORK/variant
        folder.mkdir(parents=True,exist_ok=True)
        for name in names:
            data = (REPO/name).read_bytes() if variant != 'whole' and name == 'sqlite-vec.c' else subprocess.check_output(['git','show',BASE+':'+name],cwd=REPO)
            if variant == 'selective' and name == 'sqlite-vec.c':
                assert data.count(POLICY.encode()) == 1
                data = data.replace(POLICY.encode(),b'',1)
            (folder/name).write_bytes(data)
        shutil.copyfile(REPO/'sqlite-vec.h',folder/'sqlite-vec.h')
        subprocess.run(['cc','-O3','-fPIC','-shared','-mcpu=apple-m1','-DSQLITE_VEC_ENABLE_NEON','-I'+str(REPO/'vendor'),str(folder/'sqlite-vec.c'),'-o',str(folder/'vec0.dylib')],check=True)
        manifests[variant] = {p.name:sha(p.read_bytes()) for p in folder.iterdir() if p.suffix in ['.c','.h','.dylib']}
    return manifests

def prepare(datasets):
    for dataset in datasets:
        path = WORK/(dataset+'.db')
        if path.exists():
            path.unlink()  # This experiment's generated database only.
        x,_,bucket,band,order = inputs(dataset)
        db = connect(dataset,'whole')
        db.execute(f'create virtual table t using vec0(v float[{x.shape[1]}])')
        db.execute('create table scope(id integer primary key,bucket integer,band integer)')
        db.execute('create index sb on scope(bucket)'); db.execute('create index sd on scope(band)')
        db.execute('begin')
        db.executemany('insert into t(rowid,v) values (?,?)',((int(i),x[i].tobytes()) for i in order))
        db.executemany('insert into scope values (?,?,?)',((int(i),int(bucket[i]),int(band[i])) for i in order))
        db.commit(); db.close()

def summary(runs):
    groups = {}
    equality = {}
    for r in runs:
        for p in r['points']:
            key = (r['dataset'],p['case'],p['query'])
            if key in equality:
                assert equality[key] == p['rows']
            equality[key] = p['rows']
            groups.setdefault((r['dataset'],p['case'],r['variant'],'all'),[]).append(p['ms'])
            if p['query'] >= 30:
                groups.setdefault((r['dataset'],p['case'],r['variant'],'holdout30'),[]).append(p['ms'])
    return [dict(dataset=d,case=c,variant=v,sample=s,p50_ms=statistics.median(ms),calls=len(ms)) for (d,c,v,s),ms in sorted(groups.items())]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',nargs=3)
    parser.add_argument('--queries',type=int,default=100)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--variants',nargs='+',choices=['whole','selective','adaptive'],default=['whole','selective','adaptive'])
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0],args.worker[1],int(args.worker[2])); return
    assert 1 <= args.queries <= 100 and args.rounds > 0
    prior = json.loads((PREVIOUS/'recorded/inputs.json').read_text())
    for dataset, info in prior['datasets'].items():
        for name,digest in info['hashes'].items():
            assert sha((PREVIOUS/'.work'/dataset/name).read_bytes()) == digest,(dataset,name)
    manifests = build(args.variants)
    datasets = ['sift50k','scifact','tiny2']
    prepare(datasets)
    runs, truth = [], {}
    for dataset in datasets:
        for repeat in range(args.rounds):
            for variant in (args.variants if repeat%2 == 0 else args.variants[::-1]):
                result = subprocess.run([sys.executable,__file__,'--worker',dataset,variant,str(args.queries)],capture_output=True,text=True,check=True)
                r = json.loads(result.stdout)
                oracle = r.pop('truth')
                if dataset in truth:
                    assert truth[dataset] == oracle
                truth[dataset] = oracle
                r['round'] = repeat; runs.append(r)
    result = dict(baseline=BASE,script_sha256=sha(Path(__file__).read_bytes()),libraries=manifests,queries=args.queries,rounds=args.rounds,
                  sqlite_version=sqlite3.sqlite_version,python=sys.version,platform=platform.platform(),numpy=np.__version__,
                  compiler=subprocess.check_output(['cc','--version'],text=True).splitlines()[0],
                  prior_inputs_sha256=sha((PREVIOUS/'recorded/inputs.json').read_bytes()),
                  prior_oracle_sha256=sha((PREVIOUS/'recorded/oracle.json.gz').read_bytes()),
                  runs=runs,truth=truth,summary=summary(runs))
    (HERE/'recorded/results.json.gz').write_bytes(gzip.compress(json.dumps(result).encode(),mtime=0))
    print(json.dumps(result['summary'],indent=2))

if __name__ == '__main__':
    main()
