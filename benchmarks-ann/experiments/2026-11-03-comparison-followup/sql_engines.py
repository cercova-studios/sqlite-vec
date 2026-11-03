"""Separate Turso Rust scans from libSQL scan/ANN capabilities and score accuracy."""
from pathlib import Path
import gzip,hashlib,importlib,importlib.metadata,json,resource,sys,tempfile,time
import numpy as np
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'2026-11-03-engine-comparison'
def main(engine,dataset):
    module=importlib.import_module(engine);version=importlib.metadata.version('pyturso' if engine=='turso' else 'libsql')
    data=PREVIOUS/'.work'/dataset;x=np.load(data/'vectors.npy',mmap_mode='r');q=np.load(data/'queries.npy');order=np.load(data/'order.npy');bucket=np.load(data/'bucket.npy');band=np.load(data/'band.npy');truth=json.loads((data/'truth.json').read_text())
    output={'engine':engine,'version':version,'dataset':dataset,'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'quality':[],'timings':[],'capabilities':{}}
    with tempfile.TemporaryDirectory(prefix='turso-comparison-') as root:
        start=time.perf_counter();db=module.connect(str(Path(root)/'docs.db'))
        output['sqlite_version']=db.execute('select sqlite_version()').fetchone()[0]
        calibration=db.execute("select vector_distance_l2(vector32('[0,0]'),vector32('[3,4]'))").fetchone()[0]
        output['calibration']={'expected_l2':5.,'reported':calibration,'accurate_to_2e_5':abs(calibration-5.)<=2e-5*5+2e-5}
        db.execute(f'create table docs(id integer primary key,v F32_BLOB({x.shape[1]}),bucket integer,band integer)')
        db.execute('begin')
        for i in order:db.execute('insert into docs values (?,vector32(?),?,?)',[int(i),x[i].tobytes(),int(bucket[i]),int(band[i])])
        db.commit();db.execute('create index sb on docs(bucket)');db.execute('create index sd on docs(band)');db.commit()
        try:
            db.execute("create index vi on docs(libsql_vector_idx(v,'metric=l2','max_neighbors=32','search_l=128','insert_l=200'))");db.commit();output['capabilities']['ann_index']=True
        except Exception as e:output['capabilities']['ann_index']=False;output['capabilities']['ann_error']=str(e)
        if output['capabilities']['ann_index']:
            try:db.execute("select * from vector_top_k('vi',vector32(?),10,'bucket<100')",[q[0].tobytes()]).fetchall()
            except Exception as e:output['capabilities']['dynamic_ann_filter']=False;output['capabilities']['filter_error']=str(e)
            else:output['capabilities']['dynamic_ann_filter']=True
        output['setup_seconds']=time.perf_counter()-start
        def search(query,case,ann=False):
            if ann:
                return db.execute("select d.id,vector_distance_l2(d.v,vector32(?)) distance from vector_top_k('vi',vector32(?),10) r join docs d on d.id=r.id order by distance,d.id",[query.tobytes(),query.tobytes()]).fetchall()
            condition='' if case['threshold']==1000 else f"where {case['field']} < {case['threshold']}"
            return db.execute(f'select id,vector_distance_l2(v,vector32(?)) distance from docs {condition} order by distance,id limit 10',[query.tobytes()]).fetchall()
        def validate(rows,i,case):
            ids=[r[0] for r in rows];t=case['queries'][i];k=len(t['ids'])
            assert len(ids)==len(set(ids)) and len(ids)<=k
            assert all(0<=j<len(x) and (bucket if case['field']=='bucket' else band)[j]<case['threshold'] for j in ids)
            distances=np.sqrt(np.sum((x[ids].astype('float64')-q[i].astype('float64'))**2,axis=1)) if ids else np.array([])
            actual=np.asarray([r[1] for r in rows]);assert np.isfinite(actual).all()
            accurate=bool(np.allclose(actual,distances,rtol=2e-5,atol=2e-5))
            error=float(np.max(np.abs(actual-distances)/np.maximum(distances,1e-20))) if ids else 0.
            if engine=='libsql':assert accurate,(rows,distances)
            d2=distances**2
            # Squaring a square root can perturb cutoff equality; recompute squared distances directly.
            if ids:d2=np.sum((x[ids].astype('float64')-q[i].astype('float64'))**2,axis=1)
            recall=(len(set(ids)&set(t['closer']))+min(int(np.count_nonzero(d2==t['kth'])),k-len(t['closer'])))/k if k else 1.
            return recall,accurate,error
        for name,case in truth.items():
            modes=['scan','ann'] if name=='all' and output['capabilities']['ann_index'] else ['scan']
            for mode in modes:
                search(q[0],case,mode=='ann')
                for i,query in enumerate(q):
                    rows=search(query,case,mode=='ann');recall,accurate,error=validate(rows,i,case)
                    if mode=='scan' and engine=='libsql':assert recall==1.
                    output['quality'].append({'case':name,'mode':mode,'query':i,'rows':rows,'recall':recall,'distance_accurate':accurate,'max_relative_distance_error':error})
                for repeat in range(3):
                    for i in np.random.default_rng(repeat).permutation(len(q)):
                        before=time.perf_counter_ns();rows=search(q[i],case,mode=='ann');elapsed=(time.perf_counter_ns()-before)/1e6;recall,accurate,error=validate(rows,int(i),case)
                        output['timings'].append({'case':name,'mode':mode,'query':int(i),'round':repeat,'ms':elapsed,'recall':recall,'distance_accurate':accurate})
                print(engine,dataset,name,mode,'complete',file=sys.stderr,flush=True)
        db.close();output['disk_bytes']=sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file());output['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (HERE/'recorded'/f'{engine}-{dataset}.json.gz').write_bytes(gzip.compress(json.dumps(output).encode(),mtime=0))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
