from pathlib import Path
import argparse,hashlib,json,os,platform,random,resource,sqlite3,statistics,subprocess,sys,time,tempfile
import numpy as np
ROOT=Path('/tmp/sqlite-vec-alpha-experiment');EFS=[32,64,128,256];ROUNDS=5

def quality(data,queries,truth,kth,rows,metric):
    ids=np.array([[r[0] for r in result] for result in rows],dtype=np.int64)
    reported=np.array([[r[1] for r in result] for result in rows],dtype=np.float64)
    assert ids.shape==(len(queries),10)
    assert all(len(set(row))==10 for row in ids)
    assert ids.min()>=0 and ids.max()<len(data)
    assert np.isfinite(reported).all() and (np.diff(reported,axis=1)>=-1e-7).all()
    retrieved=data[ids].astype(np.float64);q=queries.astype(np.float64)
    if metric=='cosine':
        dot=np.einsum('qd,qkd->qk',q,retrieved)
        norm=np.sqrt(np.einsum('qd,qd->q',q,q))[:,None]*np.sqrt(np.einsum('qkd,qkd->qk',retrieved,retrieved))
        distances=np.maximum(1-dot/norm,0);expected=distances
    else:
        delta=retrieved-q[:,None,:];distances=np.einsum('qkd,qkd->qk',delta,delta);expected=np.sqrt(distances)
    assert np.allclose(reported,expected,rtol=2e-5,atol=2e-5),(float(np.max(np.abs(reported-expected))),metric)
    strict=[len(set(found)&set(want))/10 for found,want in zip(ids,truth)]
    tied=np.mean(distances<=kth[:,None]+1e-10*np.maximum(1,kth[:,None]),axis=1)
    return {'recall_at_10':float(np.mean(strict)),'tie_aware_recall_at_10':float(tied.mean()),'per_query_recall':strict,'per_query_tie_aware_recall':tied.tolist(),'distances_checked':ids.size,'max_distance_error':float(np.max(np.abs(reported-expected)))}

def job(dataset,alpha,seed):
    name=f'{dataset}-alpha-{alpha:g}-seed-{seed}'
    path=ROOT/'results'/f'{name}.json'
    manifest=json.loads((ROOT/'data/manifest.json').read_text());info=manifest['datasets'][dataset]
    builds=json.loads((ROOT/'builds/manifest.json').read_text());build=builds['variants'][f'alpha-{alpha:g}']
    data=np.load(ROOT/'data'/dataset/'train.npy',mmap_mode='r');queries=np.load(ROOT/'data'/dataset/'queries.npy')
    truth=np.load(ROOT/'data'/dataset/'truth.npy');kth=np.load(ROOT/'data'/dataset/'kth-distance.npy')
    query_blobs=[q.tobytes() for q in queries];order=np.random.default_rng(seed).permutation(len(data))
    result={'dataset':dataset,'alpha':alpha,'seed':seed,'rows':len(data),'queries':len(queries),'dimensions':data.shape[1],'metric':info['metric'],'M':16,'ef_construction':200,'cache_mb':256,'k':10,'timing_rounds':ROUNDS,'library_sha256':build['library_sha256'],'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'insertion_order_sha256':hashlib.sha256(order.tobytes()).hexdigest(),'configurations':{},'timing_scope':'warm sequential single-client SQL including Python sqlite3/fetch overhead; in-memory SQLite; no affinity pinning'}
    db=sqlite3.connect(':memory:',isolation_level=None);db.enable_load_extension(True);db.load_extension(build['library']);db.enable_load_extension(False)
    db.execute(f"create virtual table h using vec0(v float[{data.shape[1]}] distance_metric={info['metric']} indexed by hnsw(m=16,ef_construction=200,ef_search=64,cache_mb=256))")
    start=time.perf_counter();cpu=time.process_time();db.execute('begin')
    db.executemany('insert into h(rowid,v) values (?,?)',((int(i),data[i].tobytes()) for i in order));db.execute('commit')
    result['build_seconds']=time.perf_counter()-start;result['build_cpu_seconds']=time.process_time()-cpu
    assert db.execute('select count(*) from h_hnsw_nodes00').fetchone()[0]==len(data)
    result['stored_graph']=dict(zip(['edge_rows','neighbor_ids','mean_degree'],db.execute('select count(*),sum(length(neighbors))/8.0,avg(length(neighbors))/8.0 from h_hnsw_edges00').fetchone()))
    print(json.dumps({'event':'built','dataset':dataset,'alpha':alpha,'seed':seed,'seconds':result['build_seconds']}),flush=True)
    sql='select rowid,distance from h where v match ? and k=10'
    for ef in EFS:
        db.execute(f"insert into h(h) values ('ef_search={ef}')")
        got=[db.execute(sql,(q,)).fetchall() for q in query_blobs]
        result['configurations'][str(ef)]={**quality(data,queries,truth,kth,got,info['metric']),'rounds':[]}
    for repetition in range(ROUNDS):
        schedule=EFS.copy();random.Random(10000+seed*100+repetition).shuffle(schedule)
        query_order=np.random.default_rng(50000+seed*100+repetition).permutation(len(queries)).tolist()
        for ef in schedule:
            db.execute(f"insert into h(h) values ('ef_search={ef}')")
            for qi in query_order[:100]:db.execute(sql,(query_blobs[qi],)).fetchall()
            lat=[];start=time.perf_counter();cpu=time.process_time()
            for qi in query_order:
                t=time.perf_counter_ns();rows=db.execute(sql,(query_blobs[qi],)).fetchall();elapsed=time.perf_counter_ns()-t
                assert len(rows)==10
                lat.append(elapsed/1000)
            wall=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu
            result['configurations'][str(ef)]['rounds'].append({'round':repetition,'seconds':wall,'cpu_seconds':cpu_seconds,'qps':len(queries)/wall,'p50_us':float(np.median(lat)),'p95_us':float(np.percentile(lat,95)),'p99_us':float(np.percentile(lat,99)),'latencies_us':lat})
    for ef in EFS:
        cfg=result['configurations'][str(ef)];cfg['median_qps']=statistics.median(r['qps'] for r in cfg['rounds']);cfg['median_p95_us']=statistics.median(r['p95_us'] for r in cfg['rounds'])
    result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
    db.close();temporary=path.with_suffix('.json.part');temporary.write_text(json.dumps(result));temporary.replace(path)
    print(json.dumps({'event':'finished','dataset':dataset,'alpha':alpha,'seed':seed,'build_seconds':result['build_seconds'],'points':{ef:{'recall':cfg['recall_at_10'],'qps':cfg['median_qps']} for ef,cfg in result['configurations'].items()}}),flush=True)

def validate():
    builds=json.loads((ROOT/'builds/manifest.json').read_text())
    rng=np.random.default_rng(0);small=rng.normal(size=(1000,32)).astype(np.float32);queries=rng.normal(size=(20,32)).astype(np.float32)
    signatures=[]
    for lib in ['/Users/rohit/sandbox/sqlite-vec/dist/vec0.dylib',builds['variants']['alpha-1']['library']]:
        db=sqlite3.connect(':memory:',isolation_level=None);db.enable_load_extension(True);db.load_extension(lib)
        db.execute('create virtual table h using vec0(v float[32] indexed by hnsw(m=16,ef_construction=200))');db.execute('begin')
        db.executemany('insert into h(rowid,v) values (?,?)',((i,row.tobytes()) for i,row in enumerate(small)));db.execute('commit')
        digest=hashlib.sha256()
        for row in db.execute('select rowid,level,neighbors from h_hnsw_edges00 order by rowid,level'):
            digest.update(str(row[:2]).encode());digest.update(row[2])
        answers=[db.execute('select rowid,distance from h where v match ? and k=10',(q.tobytes(),)).fetchall() for q in queries]
        signatures.append((digest.hexdigest(),answers));db.close()
    assert signatures[0]==signatures[1]
    checks={'current_build_graph_and_queries_equal_baseline':True,'graph_sha256':signatures[0][0],'flat_oracle_checks':{}}
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for dataset,info in manifest['datasets'].items():
        data=np.load(ROOT/'data'/dataset/'train.npy',mmap_mode='r');queries=np.load(ROOT/'data'/dataset/'queries.npy')[:10];truth=np.load(ROOT/'data'/dataset/'truth.npy')[:10];kth=np.load(ROOT/'data'/dataset/'kth-distance.npy')[:10]
        db=sqlite3.connect(':memory:',isolation_level=None);db.enable_load_extension(True);db.load_extension(builds['variants']['alpha-1']['library'])
        db.execute(f"create virtual table f using vec0(v float[{data.shape[1]}] distance_metric={info['metric']})");db.execute('begin');db.executemany('insert into f(rowid,v) values (?,?)',((i,row.tobytes()) for i,row in enumerate(data)));db.execute('commit')
        rows=[db.execute('select rowid,distance from f where v match ? and k=10',(q.tobytes(),)).fetchall() for q in queries]
        result=quality(data,queries,truth,kth,rows,info['metric'])
        assert result['tie_aware_recall_at_10']==1.0,result
        checks['flat_oracle_checks'][dataset]=result;db.close()
    (ROOT/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps({'validation':'passed'}),flush=True)

def sweep():
    datasets=list(json.loads((ROOT/'data/manifest.json').read_text())['datasets'])
    tasks=[(dataset,alpha,seed) for seed in [0,1,2] for dataset in datasets for alpha in [1.0,1.05,1.1,1.2]]
    random.Random(42).shuffle(tasks)
    first=(datasets[0],1.0,0);tasks.remove(first);tasks.insert(0,first)
    (ROOT/'schedule.json').write_text(json.dumps(tasks,indent=2))
    for index,(dataset,alpha,seed) in enumerate(tasks):
        output=ROOT/'results'/f'{dataset}-alpha-{alpha:g}-seed-{seed}.json'
        if output.exists():
            prior=json.loads(output.read_text());assert prior['harness_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            continue
        print(json.dumps({'event':'dispatch','job':index+1,'jobs':len(tasks),'dataset':dataset,'alpha':alpha,'seed':seed}),flush=True)
        result=subprocess.run([sys.executable,__file__,'job','--dataset',dataset,'--alpha',str(alpha),'--seed',str(seed)],cwd=ROOT,timeout=600)
        assert result.returncode==0,(dataset,alpha,seed,result.returncode)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['validate','sweep','job']);parser.add_argument('--dataset');parser.add_argument('--alpha',type=float);parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.command=='validate':validate()
    elif args.command=='sweep':sweep()
    else:job(args.dataset,args.alpha,args.seed)
