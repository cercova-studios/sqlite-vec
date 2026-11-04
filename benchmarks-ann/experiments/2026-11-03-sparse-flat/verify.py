"""Verify paired rows, frozen eligibility/tie oracles and archive summaries."""
from pathlib import Path
import argparse, copy, gzip, hashlib, json, math, statistics

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'2026-11-03-engine-comparison'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate(result):
    assert result['script_sha256'] == sha(HERE/'bench.py')
    assert result['prior_inputs_sha256'] == sha(PREVIOUS/'recorded/inputs.json')
    assert result['prior_oracle_sha256'] == sha(PREVIOUS/'recorded/oracle.json.gz')
    prior = json.loads(gzip.decompress((PREVIOUS/'recorded/oracle.json.gz').read_bytes()))
    groups, rows = {}, {}
    seen = set()
    for run in result['runs']:
        dataset,variant,repeat = run['dataset'],run['variant'],run['round']
        assert dataset in ['sift50k','scifact','tiny2'] and variant in ['whole','selective','adaptive']
        assert 0 <= repeat < result['rounds'] and (dataset,variant,repeat) not in seen
        seen.add((dataset,variant,repeat))
        assert len(run['points']) == 8*result['queries']
        points = set()
        for p in run['points']:
            case,i = p['case'],p['query']
            assert 0 <= i < result['queries'] and (case,i) not in points
            points.add((case,i))
            truth = result['truth'][dataset][case][i]
            ids = [r[0] for r in p['rows']]
            assert len(ids) == len(set(ids)) == truth['k']
            oracle = prior['sift50k' if dataset == 'tiny2' else dataset]
            field = 'band' if case == 'band1' else 'bucket'
            threshold = dict(all=1000,random50=500,random10=100,random5=50,random1=10,random01=1,band1=10,empty=0)[case]
            assert all(type(j) is int and 0 <= j < len(oracle[field]) and oracle[field][j] < threshold for j in ids)
            assert set(truth['closer']).issubset(ids) and set(ids).issubset(set(truth['closer'])|set(truth['ties']))
            assert all(math.isfinite(d) for _,d in p['rows']) and math.isfinite(p['ms']) and p['ms'] >= 0
            if dataset != 'tiny2' and case in oracle['truth']:
                old = oracle['truth'][case]['queries'][i]
                assert set(truth['closer']) == set(old['closer']) and set(truth['ties']) == set(old['ties'])
            key = dataset,case,i
            if key in rows:
                assert rows[key] == p['rows']
            rows[key] = p['rows']
            for sample in (['all','holdout30'] if i>=30 else ['all']):
                groups.setdefault((dataset,case,variant,sample),[]).append(p['ms'])
    assert len(seen) == 9*result['rounds']
    summary = [dict(dataset=d,case=c,variant=v,sample=s,p50_ms=statistics.median(ms),calls=len(ms)) for (d,c,v,s),ms in sorted(groups.items())]
    assert result['summary'] == summary
    for variant in ['whole','selective','adaptive']:
        for name,digest in result['libraries'][variant].items():
            if name not in ['sqlite-vec.c','vec0.dylib']:
                assert digest == result['libraries']['whole'][name]
    assert result['libraries']['adaptive']['sqlite-vec.c'] == sha(HERE.parents[2]/'sqlite-vec.c')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    args = parser.parse_args()
    result = json.loads(gzip.decompress((HERE/'recorded/results.json.gz').read_bytes()))
    validate(result)
    for field,value in [('rows',[[-1,0]]),('query',-1)]:
        bad = copy.deepcopy(result)
        bad['runs'][0]['points'][0][field] = value
        try:
            validate(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corruption accepted: '+field)
    diagnostic = json.loads(gzip.decompress((HERE/'recorded/reads.json.gz').read_bytes()))
    for dataset in ['sift50k','scifact','tiny2']:
        for variant in ['whole','selective','adaptive']:
            d = diagnostic[dataset+'-'+variant]
            assert d['rows'] == diagnostic[dataset+'-whole']['rows']
            for case,sizes in d['reads'].items():
                assert all(type(n) is int and n > 0 for n in sizes)
                run = next(r for r in result['runs'] if r['dataset'] == dataset and r['variant'] == variant)
                assert d['rows'][case] == next(p['rows'] for p in run['points'] if p['case'] == case and p['query'] == 0)
            assert d['reads']['empty'] == []
    path = HERE/'recorded/hashes.json'
    if args.write:
        files = sorted(p for p in HERE.rglob('*') if p.is_file() and '.work' not in p.parts and '__pycache__' not in p.parts and p != path)
        path.write_text(json.dumps({str(p.relative_to(HERE)):sha(p) for p in files},indent=2)+'\n')
    else:
        for name,digest in json.loads(path.read_text()).items():
            assert sha(HERE/name) == digest,name
    print(f'Verified {9*result["rounds"]*8*result["queries"]} timed queries, exact/tie-aware parity, diagnostics and corruption rejection')

if __name__ == '__main__':
    main()
