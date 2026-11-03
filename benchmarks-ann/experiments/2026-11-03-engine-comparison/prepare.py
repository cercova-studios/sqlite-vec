"""Freeze committed C sources and deterministic real-data subsets/oracles."""
from pathlib import Path
import hashlib,json,shutil,subprocess
import numpy as np
import h5py
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
WORK=HERE/'.work'
CASES=[('all','bucket',1000),('random10','bucket',100),('random1','bucket',10),('band1','band',10),('empty','bucket',0)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    build=WORK/'build';build.mkdir(exist_ok=True)
    sources={}
    for name in subprocess.check_output(['git','ls-files','sqlite-vec*.c','sqlite-vec*.h'],cwd=REPO,text=True).splitlines():
        data=subprocess.check_output(['git','show','HEAD:'+name],cwd=REPO)
        (build/name).write_bytes(data);sources[name]=hashlib.sha256(data).hexdigest()
    shutil.copyfile(REPO/'sqlite-vec.h',build/'sqlite-vec.h')
    sources['sqlite-vec.h']=sha(build/'sqlite-vec.h')
    subprocess.run(['cc','-O3','-fPIC','-shared','-mcpu=apple-m1','-DSQLITE_VEC_ENABLE_NEON','-I'+str(REPO/'vendor'),str(build/'sqlite-vec.c'),'-o',str(build/'vec0.dylib')],check=True)
    prior=HERE.parent/'2026-10-26-scifact-hybrid'
    manifest=json.loads((prior/'recorded/data-manifest.json').read_text())
    for name in ['documents.npy','queries.npy']:assert sha(prior/'.work'/name)==manifest['files'][name]
    with h5py.File(WORK/'sift.hdf5') as f:
        train_ids=np.sort(np.random.default_rng(42).choice(len(f['train']),50000,replace=False))
        sift=np.asarray(f['train'][train_ids],dtype='float32');sq=np.asarray(f['test'][:100],dtype='float32')
    inputs={'scifact':(np.load(prior/'.work/documents.npy'),np.load(prior/'.work/queries.npy')[np.random.default_rng(42).permutation(300)[:100]]),'sift50k':(sift,sq)}
    provenance={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'sources':sources,'library_sha256':sha(build/'vec0.dylib'),'sift_url':'https://ann-benchmarks.com/sift-128-euclidean.hdf5','sift_sha256':sha(WORK/'sift.hdf5'),'scifact_manifest':manifest,'datasets':{}}
    for name,(x,q) in inputs.items():
        directory=WORK/name;directory.mkdir(exist_ok=True)
        np.save(directory/'vectors.npy',x);np.save(directory/'queries.npy',q)
        n=len(x);rng=np.random.default_rng(7)
        bucket=np.empty(n,dtype='int32');bucket[rng.permutation(n)]=np.arange(n)*1000//n
        band=np.empty(n,dtype='int32');band[np.argsort(x[:,0],kind='stable')]=np.arange(n)*1000//n
        np.save(directory/'bucket.npy',bucket);np.save(directory/'band.npy',band)
        order=np.random.default_rng(19).permutation(n);np.save(directory/'order.npy',order)
        oracle={}
        for case,field,threshold in CASES:
            eligible=np.flatnonzero((bucket if field=='bucket' else band)<threshold)
            points=[]
            for query in q:
                distances=np.sum((x[eligible].astype('float64')-query.astype('float64'))**2,axis=1)
                positions=np.lexsort((eligible,distances))[:10]
                ids=eligible[positions].tolist()
                points.append({'ids':ids,'kth':float(distances[positions[-1]]) if len(ids) else None,'closer':eligible[distances < distances[positions[-1]]].tolist() if len(ids) else []})
            oracle[case]={'field':field,'threshold':threshold,'eligible':len(eligible),'queries':points}
        (directory/'truth.json').write_text(json.dumps(oracle))
        provenance['datasets'][name]={'rows':n,'dimensions':x.shape[1],'queries':len(q),'hashes':{p.name:sha(p) for p in directory.iterdir() if p.is_file()}}
    (HERE/'recorded/inputs.json').write_text(json.dumps(provenance,indent=2)+'\n')
if __name__=='__main__':main()
