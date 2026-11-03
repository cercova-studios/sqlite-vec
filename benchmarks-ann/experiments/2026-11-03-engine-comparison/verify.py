"""Recompute correctness/recall summaries from receipts and frozen masks/oracles."""
from pathlib import Path
import argparse,gzip,hashlib,json,math,statistics
HERE=Path(__file__).resolve().parent
ENGINES=['upstream-flat','fork-flat','fork-hnsw','zvec-hnsw','lance-exact','lance-ann','lance-hnsw1']
DATASETS=['scifact','sift50k']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def capture():
    import numpy as np
    archive={}
    for dataset in DATASETS:
        path=HERE/'.work'/dataset
        x=np.load(path/'vectors.npy');q=np.load(path/'queries.npy');bucket=np.load(path/'bucket.npy');band=np.load(path/'band.npy')
        truth=json.loads((path/'truth.json').read_text())
        for case in truth.values():
            mask=(bucket if case['field']=='bucket' else band)<case['threshold'];ids=np.flatnonzero(mask)
            for i,point in enumerate(case['queries']):
                d2=np.sum((x[ids].astype('float64')-q[i].astype('float64'))**2,axis=1)
                point['ties']=ids[d2==point['kth']].tolist() if point['kth'] is not None else []
        archive[dataset]={'bucket':bucket.tolist(),'band':band.tolist(),'truth':truth}
    (HERE/'recorded/oracle.json.gz').write_bytes(gzip.compress(json.dumps(archive).encode(),mtime=0))
def recall(ids,truth):
    k=len(truth['ids'])
    if not k:return 1.,1.
    strict=len(set(ids)&set(truth['ids']))/k
    tied=(len(set(ids)&set(truth['closer']))+min(len(set(ids)&set(truth['ties'])),k-len(truth['closer'])))/k
    return strict,tied

def self_test():
    truth={'ids':[0,1,2],'closer':[0,1],'ties':[2,3,4]}
    assert recall([0,1,4],truth)==(2/3,1.)
    assert recall([0,2,3],truth)==(2/3,2/3)
    assert recall([2,3,4],truth)==(1/3,1/3)
    assert recall([],{'ids':[],'closer':[],'ties':[]})==(1.,1.)

def validate(result,oracle):
    expected_configs=[32,128,512] if result['engine'] in ['fork-hnsw','zvec-hnsw','lance-ann','lance-hnsw1'] else [0]
    assert len(result['quality'])==len(expected_configs)*500
    assert len(result['timings'])==len(expected_configs)*1500
    seen=set()
    for point in result['quality']:
        key=(point['config'],point['case'],point['query']);assert key not in seen;seen.add(key)
        assert point['config'] in expected_configs and 0<=point['query']<100
        case=oracle['truth'][point['case']];truth=case['queries'][point['query']];k=len(truth['ids'])
        ids=[r[0] for r in point['rows']]
        assert len(set(ids))==len(ids) and len(ids)<=k
        assert all(type(i) is int and 0<=i<result['rows'] and oracle[case['field']][i]<case['threshold'] for i in ids)
        assert all(math.isfinite(r[1]) for r in point['rows'])
        strict,tied=recall(ids,truth)
        assert strict==point['strict_recall'] and tied==point['tie_recall']
        if result['engine'] in ['upstream-flat','fork-flat','lance-exact']:assert tied==1.
    quality={(p['config'],p['case'],p['query']):p['tie_recall'] for p in result['quality']}
    seen=set()
    for point in result['timings']:
        key=(point['config'],point['case'],point['round'],point['query']);assert key not in seen;seen.add(key)
        assert point['config'] in expected_configs and point['case'] in oracle['truth'] and 0<=point['round']<3 and 0<=point['query']<100
        assert math.isfinite(point['ms']) and point['ms']>=0 and 0<=point['recall']<=1
        assert point['recall']==quality[point['config'],point['case'],point['query']]
    summary=[]
    for config in expected_configs:
        for case in oracle['truth']:
            points=[p for p in result['timings'] if p['config']==config and p['case']==case]
            latency=sorted(p['ms'] for p in points)
            summary.append({'dataset':result['dataset'],'engine':result['engine'],'config':config,'case':case,'recall':statistics.mean(p['recall'] for p in points),'min_query_recall':min(p['recall'] for p in points),'p50_ms':statistics.median(latency),'p95_ms':latency[math.ceil(.95*len(latency))-1]})
    return summary
def summarize():
    oracle=json.loads(gzip.decompress((HERE/'recorded/oracle.json.gz').read_bytes()))
    runtime=json.loads((HERE/'recorded/runtime.json').read_text());inputs=json.loads((HERE/'recorded/inputs.json').read_text())
    assert runtime['worker_sha256']==sha(HERE/'worker.py')
    rows=[];builds=[]
    for dataset in DATASETS:
        for engine in ENGINES:
            result=json.loads(gzip.decompress((HERE/'recorded'/f'{dataset}-{engine}.json.gz').read_bytes()))
            assert result['engine']==engine and result['dataset']==dataset and result['source_hash']==runtime['worker_sha256']
            assert result['rows']==inputs['datasets'][dataset]['rows']
            if engine.startswith('fork'):assert result['library_sha256']==inputs['library_sha256']
            rows.extend(validate(result,oracle[dataset]))
            builds.append({k:result[k] for k in ['dataset','engine','version','build_seconds','disk_bytes','peak_process_rss_bytes','distance_units','settings']})
    matched=[]
    for dataset in DATASETS:
        for engine in ENGINES:
            for case in ['all','random10','random1','band1']:
                for target in [.95,.99]:
                    eligible=[r for r in rows if r['dataset']==dataset and r['engine']==engine and r['case']==case and r['recall']>=target]
                    best=min(eligible,key=lambda r:r['p50_ms']) if eligible else None
                    matched.append({'dataset':dataset,'engine':engine,'case':case,'target':target,'best':best})
    return {'points':rows,'matched':matched,'builds':builds}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture',action='store_true');parser.add_argument('--write',action='store_true');args=parser.parse_args()
    self_test()
    if args.capture:capture()
    result=summarize();path=HERE/'recorded/summary.json'
    if args.write:
        path.write_text(json.dumps(result,indent=2)+'\n')
        paths=[p for p in HERE.rglob('*') if p.is_file() and '.work' not in p.parts and '__pycache__' not in p.parts and p.name!='hashes.json']
        (HERE/'recorded/hashes.json').write_text(json.dumps({str(p.relative_to(HERE)):sha(p) for p in sorted(paths)},indent=2)+'\n')
    else:
        assert result==json.loads(path.read_text())
        for name,digest in json.loads((HERE/'recorded/hashes.json').read_text()).items():assert sha(HERE/name)==digest,name
    print('Verified 14 runs, 15000 quality points and 45000 timed calls')
if __name__=='__main__':main()
