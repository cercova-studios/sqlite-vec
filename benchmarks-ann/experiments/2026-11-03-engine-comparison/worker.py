"""One fresh process per engine/dataset; never silently drop invalid results."""
from pathlib import Path
import gc,gzip,hashlib,importlib.metadata,json,os,resource,sqlite3,sys,tempfile,time
import numpy as np
HERE=Path(__file__).resolve().parent
WORK=HERE/'.work'
def disk(path):return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
def run(engine,dataset):
    data=WORK/dataset
    x=np.load(data/'vectors.npy',mmap_mode='r');queries=np.load(data/'queries.npy');order=np.load(data/'order.npy')
    bucket=np.load(data/'bucket.npy');band=np.load(data/'band.npy');truth=json.loads((data/'truth.json').read_text())
    qlists=[q.tolist() for q in queries];blobs=[q.tobytes() for q in queries]
    result={'engine':engine,'dataset':dataset,'rows':len(x),'dimensions':x.shape[1],'sqlite':sqlite3.sqlite_version,'quality':[],'timings':[],'settings':{},'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with tempfile.TemporaryDirectory(prefix='vec-comparison-') as root:
        root=Path(root);start=time.perf_counter()
        if engine.startswith(('fork','upstream')):
            import sqlite_vec
            library=sqlite_vec.loadable_path() if engine=='upstream-flat' else str(WORK/'build/vec0.dylib')
            result['library_sha256']=hashlib.sha256(Path(library if library.endswith('.dylib') else library+'.dylib').read_bytes()).hexdigest()
            db=sqlite3.connect(root/'db.sqlite',isolation_level=None);db.enable_load_extension(True);db.load_extension(library);db.enable_load_extension(False)
            result['version']=db.execute('select vec_version()').fetchone()[0]
            ann=' indexed by hnsw(m=16,ef_construction=200,ef_search=32,cache_mb=256)' if engine=='fork-hnsw' else ''
            db.execute(f'create virtual table t using vec0(v float[{x.shape[1]}]{ann})')
            db.execute('create table scope(id integer primary key,bucket integer,band integer)')
            db.execute('begin');db.executemany('insert into t(rowid,v) values (?,?)',((int(i),x[i].tobytes()) for i in order))
            db.executemany('insert into scope values (?,?,?)',((int(i),int(bucket[i]),int(band[i])) for i in order));db.execute('commit')
            db.execute('create index scope_bucket on scope(bucket)');db.execute('create index scope_band on scope(band)')
            result['settings']={'m':16,'ef_construction':200,'cache_mb':256} if ann else {'strategy':'flat'}
            def configure(config):
                if ann:db.execute(f"insert into t(t) values ('ef_search={config}')")
            def search(i,case,config):
                predicate="" if case['threshold']==1000 else f" and rowid in (select id from scope where {case['field']} < ?)"
                parameters=[blobs[i]] if not predicate else [blobs[i],case['threshold']]
                rows=db.execute("select rowid,distance from t where v match ? and k=10"+predicate+" order by distance",parameters).fetchall()
                return [(int(i),float(d)) for i,d in rows]
            configs=[32,128,512] if ann else [0]
            def reopen():
                nonlocal db
                db.close();db=sqlite3.connect(root/'db.sqlite',isolation_level=None);db.enable_load_extension(True);db.load_extension(library);db.enable_load_extension(False)
            def close():db.close()
        elif engine=='zvec-hnsw':
            import zvec
            zvec.init(query_threads=1,optimize_threads=1)
            result['version']=importlib.metadata.version('zvec')
            schema=zvec.CollectionSchema(name='docs',fields=[zvec.FieldSchema(f,zvec.DataType.INT32,index_param=zvec.InvertIndexParam()) for f in ['bucket','band']],vectors=[zvec.VectorSchema('vector',zvec.DataType.VECTOR_FP32,dimension=x.shape[1],index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.L2,m=16,ef_construction=200))])
            collection=zvec.create_and_open(str(root/'zvec'),schema=schema)
            for offset in range(0,len(order),256):
                docs=[zvec.Doc(id=str(i),vectors={'vector':x[i].tolist()},fields={'bucket':int(bucket[i]),'band':int(band[i])}) for i in order[offset:offset+256]]
                statuses=collection.insert(docs)
                assert all(s.ok() for s in statuses),statuses
            collection.optimize();collection.flush()
            result['settings']={'m':16,'ef_construction':200,'query_threads':1,'optimize_threads':1,'scalar_indexes':'invert'}
            def configure(config):pass
            def search(i,case,config):
                docs=collection.query(queries=zvec.Query(field_name='vector',vector=qlists[i],param=zvec.HnswQueryParam(ef=config)),topk=10,filter=None if case['threshold']==1000 else f"{case['field']} < {case['threshold']}",output_fields=[])
                return [(int(d.id),float(d.score)) for d in docs]
            configs=[32,128,512]
            def reopen():
                nonlocal collection
                collection.close();collection=zvec.open(str(root/'zvec'))
            def close():collection.close()
        else:
            import lancedb,pyarrow as pa
            from lancedb.index import BTree,HnswFlat
            pa.set_cpu_count(1);pa.set_io_thread_count(1)
            result['version']=importlib.metadata.version('lancedb')
            db=lancedb.connect(str(root/'lance'))
            table=pa.table({'id':np.asarray(order,dtype='int64'),'bucket':bucket[order],'band':band[order],'vector':pa.FixedSizeListArray.from_arrays(pa.array(x[order].reshape(-1)),x.shape[1])})
            collection=db.create_table('docs',table);del table;gc.collect()
            collection.create_index('bucket',config=BTree());collection.create_index('band',config=BTree())
            if engine in ['lance-ann','lance-hnsw1']:collection.create_index('vector',config=HnswFlat(distance_type='l2',num_partitions=1 if engine=='lance-hnsw1' else 64,m=16,ef_construction=200))
            result['settings']={'index':'IVF_HNSW_FLAT' if engine in ['lance-ann','lance-hnsw1'] else 'exact','partitions':(1 if engine=='lance-hnsw1' else 64) if engine in ['lance-ann','lance-hnsw1'] else None,'m':16,'ef_construction':200,'arrow_cpu_threads':1,'arrow_io_threads':1,'scalar_indexes':'BTREE','lance_compute_threads':os.environ.get('LANCE_CPU_THREADS'),'lance_io_threads':os.environ.get('LANCE_IO_THREADS')}
            def configure(config):pass
            def search(i,case,config):
                query=collection.search(qlists[i]).distance_type('l2').limit(10).select(['id','_distance'])
                if case['threshold']!=1000:query=query.where(f"{case['field']} < {case['threshold']}",prefilter=True)
                if engine in ['lance-ann','lance-hnsw1']:query=query.nprobes(1 if engine=='lance-hnsw1' else {32:8,128:32,512:64}[config]).ef(config)
                else:query=query.bypass_vector_index()
                return [(int(r['id']),float(r['_distance'])) for r in query.to_list()]
            configs=[32,128,512] if engine in ['lance-ann','lance-hnsw1'] else [0]
            def reopen():
                nonlocal db,collection
                db=lancedb.connect(str(root/'lance'));collection=db.open_table('docs')
            def close():pass
        result['build_seconds']=time.perf_counter()-start;result['disk_bytes']=disk(root)
        unit=None
        def validate(rows,i,case):
            nonlocal unit
            ids=[r[0] for r in rows];expected=case['queries'][i];k=len(expected['ids'])
            assert len(ids)==len(set(ids)) and len(ids)<=k,(engine,ids,k)
            assert all(0<=j<len(x) and (bucket if case['field']=='bucket' else band)[j]<case['threshold'] for j in ids)
            d2=np.sum((np.asarray(x[ids],dtype='float64')-queries[i].astype('float64'))**2,axis=1) if ids else np.array([])
            reported=np.asarray([r[1] for r in rows]);assert np.isfinite(reported).all()
            assert np.all(np.diff(reported)>=-1e-5),reported
            if ids:
                if unit is None:
                    if np.allclose(reported,np.sqrt(d2),rtol=2e-5,atol=2e-5):unit='l2'
                    elif np.allclose(reported,d2,rtol=2e-5,atol=2e-5):unit='squared_l2'
                    else:raise AssertionError((engine,reported,d2))
                assert np.allclose(reported,np.sqrt(d2) if unit=='l2' else d2,rtol=2e-5,atol=2e-5),(engine,reported,d2)
            strict=len(set(ids)&set(expected['ids']))/k if k else 1.
            # Every true closer-than-cutoff neighbor is unique; any cutoff tie may fill the remaining slots.
            tied=(len(set(ids)&set(expected['closer']))+min(int(np.count_nonzero(d2==expected['kth'])),k-len(expected['closer'])))/k if k else 1.
            if engine in ['fork-flat','upstream-flat','lance-exact']:assert tied==1.,(engine,i,ids,expected)
            return strict,tied
        for config in configs:
            configure(config)
            for name,case in truth.items():
                search(0,case,config)
                for i in range(len(queries)):
                    rows=search(i,case,config);strict,tied=validate(rows,i,case)
                    result['quality'].append({'config':config,'case':name,'query':i,'rows':rows,'strict_recall':strict,'tie_recall':tied})
                for repeat in range(3):
                    for i in np.random.default_rng(repeat).permutation(len(queries)):
                        before=time.perf_counter_ns();rows=search(int(i),case,config);elapsed=(time.perf_counter_ns()-before)/1e6
                        strict,tied=validate(rows,int(i),case)
                        result['timings'].append({'config':config,'case':name,'query':int(i),'round':repeat,'ms':elapsed,'recall':tied})
            print(f'{dataset} {engine} config {config} complete',file=sys.stderr,flush=True)
        reopen()
        configure(configs[-1])
        for name,case in truth.items():
            for i in range(3):validate(search(i,case,configs[-1]),i,case)
        close()
        result['distance_units']=unit;result['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return result
if __name__=='__main__':
    result=run(sys.argv[1],sys.argv[2])
    (HERE/'recorded'/f'{sys.argv[2]}-{sys.argv[1]}.json.gz').write_bytes(gzip.compress(json.dumps(result).encode(),mtime=0))
