"""Count vector BLOB API bytes separately from uninstrumented timings."""
from pathlib import Path
import gzip, json, shutil, sqlite3, subprocess, sys
import bench

HERE = Path(__file__).resolve().parent
WRAPPER = '''static int bench_read(sqlite3_blob *blob, void *data, int bytes, int offset) {
  fprintf(stderr, "VREAD %d\\n", bytes);
  return sqlite3_blob_read(blob, data, bytes, offset);
}
'''

def child(dataset, variant):
    _,queries,_,_,_ = bench.inputs(dataset)
    db = sqlite3.connect(bench.WORK/(dataset+'.db'),isolation_level=None)
    db.enable_load_extension(True)
    db.load_extension(str(bench.WORK/('reads-'+variant)/'vec0.dylib'))
    db.enable_load_extension(False)
    rows = {}
    for name,field,threshold in bench.CASES:
        predicate = '' if name == 'all' else f' and rowid in (select id from scope where {field}<{threshold})'
        print('POINT',name,file=sys.stderr,flush=True)
        rows[name] = db.execute('select rowid,distance from t where v match ? and k=10'+predicate+' order by distance',[queries[0].tobytes()]).fetchall()
    db.close()
    print(json.dumps(rows))

def main():
    if len(sys.argv)>1:
        child(*sys.argv[1:]); return
    output = {}
    for variant in ['whole','selective','adaptive']:
        folder = bench.WORK/('reads-'+variant)
        shutil.copytree(bench.WORK/variant,folder,dirs_exist_ok=True)
        path = folder/'sqlite-vec.c'
        source = path.read_text()
        if variant == 'whole':
            anchor = 'rc = sqlite3_blob_read(blobVectors, baseVectors, currentBaseVectorsSize, 0);'
            assert source.count(anchor) == 1
            source = source.replace(anchor,anchor.replace('sqlite3_blob_read','bench_read'),1)
            anchor = 'int vec0Filter_knn_chunks_iter('
        else:
            anchor = 'static int vec0_read_vector_chunk('
            first,last = source.index(anchor),source.index('int vec0Filter_knn_chunks_iter(')
            region = source[first:last]
            assert region.count('sqlite3_blob_read(') == (1 if variant == 'selective' else 2)
            source = source[:first]+region.replace('sqlite3_blob_read(','bench_read(')+source[last:]
        source = source.replace(anchor,WRAPPER+'\n'+anchor,1)
        path.write_text(source)
        subprocess.run(['cc','-O3','-fPIC','-shared','-mcpu=apple-m1','-DSQLITE_VEC_ENABLE_NEON','-I'+str(bench.REPO/'vendor'),str(path),'-o',str(folder/'vec0.dylib')],check=True)
        for dataset in ['sift50k','scifact','tiny2']:
            result = subprocess.run([sys.executable,__file__,dataset,variant],check=True,capture_output=True,text=True)
            points = {}
            active = None
            for line in result.stderr.splitlines():
                fields = line.split()
                if fields[0] == 'POINT':
                    active = fields[1]; points[active] = []
                elif fields[0] == 'VREAD':
                    assert active is not None
                    points[active].append(int(fields[1]))
            output[dataset+'-'+variant] = dict(reads=points,rows=json.loads(result.stdout),source_sha256=bench.sha(path.read_bytes()))
    for dataset in ['sift50k','scifact','tiny2']:
        assert output[dataset+'-whole']['rows'] == output[dataset+'-selective']['rows'] == output[dataset+'-adaptive']['rows']
    (HERE/'recorded/reads.json.gz').write_bytes(gzip.compress(json.dumps(output).encode(),mtime=0))
    print('Verified diagnostic rows and recorded vector read sizes for three strategies')

if __name__ == '__main__':
    main()
