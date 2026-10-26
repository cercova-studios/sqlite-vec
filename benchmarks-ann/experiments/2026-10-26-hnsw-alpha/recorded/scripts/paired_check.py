from pathlib import Path
import argparse,json,random,sqlite3,subprocess,sys,time
import numpy as np
from benchmark import ROOT,EFS,quality

def worker(dataset,alpha):
    manifest=json.loads((ROOT/'builds/manifest.json').read_text());lib=manifest['variants'][f'alpha-{alpha:g}']['library']
    info=json.loads((ROOT/'data/manifest.json').read_text())['datasets'][dataset]
    data=np.load(ROOT/'data'/dataset/'train.npy',mmap_mode='r');queries=np.load(ROOT/'data'/dataset/'queries.npy');truth=np.load(ROOT/'data'/dataset/'truth.npy');kth=np.load(ROOT/'data'/dataset/'kth-distance.npy')
    blobs=[q.tobytes() for q in queries];order=np.random.default_rng(0).permutation(len(data))
    db=sqlite3.connect(':memory:',isolation_level=None);db.enable_load_extension(True);db.load_extension(lib)
    db.execute(f"create virtual table h using vec0(v float[{data.shape[1]}] distance_metric={info['metric']} indexed by hnsw(m=16,ef_construction=200,ef_search=64,cache_mb=256))")
    db.execute('begin');db.executemany('insert into h(rowid,v) values (?,?)',((int(i),data[i].tobytes()) for i in order));db.execute('commit')
    original=json.loads((ROOT/'results'/f'{dataset}-alpha-{alpha:g}-seed-0.json').read_text())
    neighbors=db.execute('select sum(length(neighbors))/8.0 from h_hnsw_edges00').fetchone()[0]
    assert neighbors==original['stored_graph']['neighbor_ids']
    sql='select rowid,distance from h where v match ? and k=10'
    for ef in EFS:
        db.execute(f"insert into h(h) values ('ef_search={ef}')")
        got=[db.execute(sql,(q,)).fetchall() for q in blobs]
        checked=quality(data,queries,truth,kth,got,info['metric'])
        assert checked['per_query_recall']==original['configurations'][str(ef)]['per_query_recall']
    print(json.dumps({'ready':True,'dataset':dataset,'alpha':alpha,'recall_and_neighbor_count_reproduced':True}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        ef=request['ef'];seed=request['seed'];db.execute(f"insert into h(h) values ('ef_search={ef}')")
        order=np.random.default_rng(seed).permutation(len(queries)).tolist()
        for qi in order[:100]:db.execute(sql,(blobs[qi],)).fetchall()
        lat=[];start=time.perf_counter();cpu=time.process_time()
        for qi in order:
            t=time.perf_counter_ns();rows=db.execute(sql,(blobs[qi],)).fetchall();elapsed=time.perf_counter_ns()-t
            assert len(rows)==10;lat.append(elapsed/1000)
        wall=time.perf_counter()-start;cpu=time.process_time()-cpu
        print(json.dumps({'alpha':alpha,'ef':ef,'round':request['round'],'seconds':wall,'cpu_seconds':cpu,'qps':len(queries)/wall,'p50_us':float(np.median(lat)),'p95_us':float(np.percentile(lat,95)),'latencies_us':lat}),flush=True)
    db.close()

def parent():
    receipts=[]
    for dataset in ['glove-100-angular','sift-128-euclidean']:
        workers={}
        try:
            for alpha in [1.,1.1]:
                p=subprocess.Popen([sys.executable,__file__,'--worker','--dataset',dataset,'--alpha',str(alpha)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
                workers[alpha]=p;message=json.loads(p.stdout.readline());assert message['ready'],message
                print(json.dumps(message),flush=True)
            for repetition in range(10):
                settings=EFS.copy();random.Random(9100+repetition).shuffle(settings)
                for ef in settings:
                    variants=[1.,1.1]
                    if (repetition+ef//32)%2:variants.reverse()
                    pair=[]
                    for alpha in variants:
                        p=workers[alpha];p.stdin.write(json.dumps({'ef':ef,'round':repetition,'seed':82000+repetition})+'\n');p.stdin.flush()
                        measured=json.loads(p.stdout.readline());assert measured['alpha']==alpha and measured['ef']==ef
                        pair.append(measured)
                    receipts.append({'dataset':dataset,'ef':ef,'round':repetition,'order':variants,'measurements':pair})
            print(json.dumps({'event':'paired_completed','dataset':dataset,'pairs':40}),flush=True)
        finally:
            for p in workers.values():
                if p.poll() is None:
                    try:p.stdin.write('{"stop":true}\n');p.stdin.flush();p.wait(timeout=10)
                    except Exception:p.kill();p.wait()
                assert p.returncode==0,p.returncode
    (ROOT/'paired_timing.json').write_text(json.dumps(receipts))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--dataset');parser.add_argument('--alpha',type=float);a=parser.parse_args()
    if a.worker:worker(a.dataset,a.alpha)
    else:parent()
