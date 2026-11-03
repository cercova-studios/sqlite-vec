"""Diagnostic-only instrumentation; never changes the working C core."""
from pathlib import Path
import gzip,hashlib,json,shutil,sqlite3,subprocess,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'2026-11-03-engine-comparison'
REPO=HERE.parents[2]
WORK=HERE/'.work'
def replace_once(text,old,new):
    assert text.count(old)==1,old[:90]
    return text.replace(old,new,1)
def build():
    source=PREVIOUS/'.work/build';target=WORK/'instrumented';shutil.copytree(source,target,dirs_exist_ok=True)
    p=target/'sqlite-vec.c';s=p.read_text();s=replace_once(s,'#include <string.h>','#include <string.h>\n#include <time.h>\nstatic unsigned long long bench_distances, bench_allowed;\nstatic unsigned long long bench_clock(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (unsigned long long)t.tv_sec*1000000000ULL+t.tv_nsec; }')
    s=replace_once(s,'struct Array **out) {\n  sqlite3_value *item;', 'struct Array **out) {\n  unsigned long long bench_start=bench_clock();\n  sqlite3_value *item;')
    s=replace_once(s,'qsort(a->z, a->length, a->element_size, _cmp);\n  return SQLITE_OK;', 'qsort(a->z, a->length, a->element_size, _cmp);\n  fprintf(stderr,"COLLECT %llu %zu\\n",bench_clock()-bench_start,a->length);\n  return SQLITE_OK;');p.write_text(s)
    p=target/'sqlite-vec-hnsw.c';s=p.read_text();s=replace_once(s,'int32_t idx, f32 *d) {\n  if (!qcode)', 'int32_t idx, f32 *d) {\n  bench_distances++;\n  if (!qcode)')
    s=replace_once(s,'static int hnsw_allowed(struct HnswCtx *x, const struct Array *a, int32_t idx) {','static int hnsw_allowed(struct HnswCtx *x, const struct Array *a, int32_t idx) {\n  bench_allowed++;')
    s=replace_once(s,'struct vec0_query_knn_data *knn) {\n  struct HnswCtx x;', 'struct vec0_query_knn_data *knn) {\n  unsigned long long bench_start=bench_clock();\n  bench_distances=bench_allowed=0;\n  struct HnswCtx x;')
    s=replace_once(s,'sqlite3_free(res);\n  hnsw_end(&x);\n  return rc;', 'sqlite3_free(res);\n  hnsw_end(&x);\n  fprintf(stderr,"HNSW %llu %llu %llu\\n",bench_clock()-bench_start,bench_distances,bench_allowed);\n  return rc;');p.write_text(s)
    subprocess.run(['cc','-O3','-fPIC','-shared','-mcpu=apple-m1','-DSQLITE_VEC_ENABLE_NEON','-I'+str(REPO/'vendor'),str(target/'sqlite-vec.c'),'-o',str(target/'vec0.dylib')],check=True)
def connect(lib):
    db=sqlite3.connect(WORK/'filter.db',isolation_level=None);db.enable_load_extension(True);db.load_extension(str(lib));db.enable_load_extension(False);return db
def child(variant):
    lib=PREVIOUS/'.work/build/vec0.dylib' if variant=='base' else WORK/'instrumented/vec0.dylib'
    db=connect(lib);q=np.load(PREVIOUS/'.work/sift50k/queries.npy')
    result=[]
    variants={'all':('',[]),'random10':(' and rowid in (select id from scope where bucket<100)',[]),'random1':(' and rowid in (select id from scope where bucket<10)',[])}
    for name,(predicate,_) in variants.items():
        sql='select rowid,distance from t where v match ? and k=10'+predicate+' order by distance'
        for query in q[:20]:db.execute(sql,[query.tobytes()]).fetchall()
        for i,query in enumerate(q[:20]):
            print('POINT',name,i,file=sys.stderr,flush=True)
            start=time.perf_counter_ns();rows=db.execute(sql,[query.tobytes()]).fetchall();elapsed=time.perf_counter_ns()-start
            result.append({'case':name,'query':i,'ns':elapsed,'rows':rows})
    print(json.dumps(result))
    db.close()
def main():
    if len(sys.argv)>1:child(sys.argv[1]);return
    build();path=WORK/'filter.db'
    if path.exists():raise RuntimeError('use a fresh work directory; filter.db already exists')
    data=PREVIOUS/'.work/sift50k';x=np.load(data/'vectors.npy');bucket=np.load(data/'bucket.npy');order=np.load(data/'order.npy')
    db=connect(PREVIOUS/'.work/build/vec0.dylib')
    db.execute('create virtual table t using vec0(v float[128] indexed by hnsw(m=16,ef_construction=200,ef_search=32,cache_mb=256))')
    db.execute('create table scope(id integer primary key,bucket integer)');db.execute('create index sb on scope(bucket)')
    db.execute('begin');db.executemany('insert into t(rowid,v) values (?,?)',((int(i),x[i].tobytes()) for i in order));db.executemany('insert into scope values (?,?)',((int(i),int(bucket[i])) for i in order));db.execute('commit');db.close()
    runs={}
    for variant in ['base','instrumented']:
        result=subprocess.run([sys.executable,__file__,variant],check=True,capture_output=True,text=True)
        (HERE/'recorded'/f'{variant}.trace').write_text(result.stderr)
        runs[variant]=json.loads(result.stdout)
    assert [p['rows'] for p in runs['base']]==[p['rows'] for p in runs['instrumented']]
    (HERE/'recorded/filter-profile.json').write_text(json.dumps(runs,indent=2)+'\n')
if __name__=='__main__':main()
