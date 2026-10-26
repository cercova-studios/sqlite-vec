from pathlib import Path
import hashlib,json,time,platform,sqlite3,subprocess,ctypes
import numpy as np
import h5py
from paths import ROOT;N=50000;Q=1000;K=10
manifest={'rows':N,'queries':Q,'k':K,'sample_seed':42,'oracle':'independent direct float64 C; validated with elementwise NumPy; BLAS avoided after non-finite matmul output','platform':platform.platform(),'numpy':np.__version__,'h5py':h5py.__version__,'datasets':{}}
for name,metric in [('glove-100-angular','cosine'),('sift-128-euclidean','l2')]:
    start=time.monotonic();path=ROOT/'data'/f'{name}.hdf5';partial=path.with_suffix('.hdf5.part')
    if not path.exists():partial.rename(path)
    digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    folder=ROOT/'data'/name;folder.mkdir(exist_ok=True)
    rng=np.random.default_rng(42)
    with h5py.File(path,'r') as h:
        print(name,{key:h[key].shape for key in h.keys()},flush=True)
        assert {'train','test','neighbors','distances'}<=set(h.keys())
        assert h['train'].shape[1]==h['test'].shape[1]
        ids=np.sort(rng.choice(len(h['train']),N,replace=False));qids=np.sort(rng.choice(len(h['test']),Q,replace=False))
        dim=h['train'].shape[1]
        data=np.lib.format.open_memmap(folder/'train.npy',mode='w+',dtype=np.float32,shape=(N,dim))
        filled=0
        for lo in range(0,len(h['train']),8192):
            selected=ids[(ids>=lo)&(ids<lo+8192)]
            if not len(selected):continue
            block=h['train'][lo:lo+8192]
            data[filled:filled+len(selected)]=block[selected-lo];filled+=len(selected)
        assert filled==N;data.flush()
        query=h['test'][qids].astype(np.float32)
        shapes={key:list(h[key].shape) for key in h.keys()}
        attrs={key:str(value) for key,value in h.attrs.items()}
    assert np.isfinite(data).all() and np.isfinite(query).all()
    np.save(folder/'queries.npy',query);np.save(folder/'train-source-ids.npy',ids);np.save(folder/'query-source-ids.npy',qids)
    oracle=ctypes.CDLL(str(ROOT/'exact_oracle.dylib')).exact_top10
    oracle.argtypes=[np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')]
    oracle.restype=ctypes.c_int
    truth=np.empty((Q,K),dtype=np.int64);kth=np.empty(Q);next_distance=np.empty(Q)
    assert oracle(data,query,N,Q,dim,int(metric=='cosine'),truth,kth,next_distance)==0
    ties=int(np.count_nonzero(np.abs(kth-next_distance)<=1e-12*np.maximum(1,kth)))
    # Validate independently using elementwise NumPy/einsum; no BLAS matmul.
    x=np.asarray(data,dtype=np.float64)
    norms=np.sqrt(np.einsum('ij,ij->i',x,x)) if metric=='cosine' else None
    for j in range(10):
        q64=query[j].astype(np.float64)
        if metric=='l2':
            delta=x-q64
            direct=np.einsum('ij,ij->i',delta,delta)
        else:
            direct=1-np.einsum('ij,j->i',x,q64)/(norms*np.sqrt((q64*q64).sum()))
        direct=np.maximum(direct,0)
        assert np.max(np.abs(direct[truth[j]]-np.sort(direct)[:K]))<1e-8
    np.save(folder/'truth.npy',truth);np.save(folder/'kth-distance.npy',kth)
    entry={'metric':metric,'dimensions':dim,'source_shapes':shapes,'hdf5_attrs':attrs,'sha256':digest,'source_url':f'https://ann-benchmarks.com/{name}.hdf5','prepared_seconds':time.monotonic()-start,'queries_tied_at_k':ties,'arrays':{f.name:hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in folder.glob('*.npy')}}
    manifest['datasets'][name]=entry
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2))
    print(name,'prepared',entry['prepared_seconds'],'seconds; tied queries',ties,flush=True)
    del x,data
