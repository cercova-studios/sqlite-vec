"""Validate raw benchmark shape and regenerate descriptive summaries."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent

def summarize(raw):
    previous = HERE.parent/'2026-10-26-scifact-hybrid/recorded/receipts.json.gz'
    frozen = json.loads(gzip.decompress(previous.read_bytes()))
    gold = {(r['scenario'],r['query_id']):[p[0] for p in r['candidates']] for r in frozen['results'] if r['method']=='keyword'}
    for q in raw['quality']:
        rows=q['rows']
        assert [r[0] for r in rows]==gold[q['scope'],q['id']]
        assert len(rows)<=50 and len({r[0] for r in rows})==len(rows)
        assert all(math.isfinite(r[1]) for r in rows)
        assert rows==sorted(rows,key=lambda r:(r[1],r[0]))
    assert len(raw['quality']) == 600
    assert len({(q['scope'],q['id']) for q in raw['quality']}) == 600
    assert len(raw['timings']) == 96 and len(raw['controls']) == 108
    summaries = []
    for key in ('full','scoped'):
        row = dict(scope=key)
        for variant in ('in','exists'):
            values = [r['ms'] for r in raw['timings'] if r['scope']==key and r['variant']==variant]
            assert len(values) == 24 and all(math.isfinite(v) and v>=0 for v in values)
            row[variant+'_p50_ms'] = statistics.median(values)
            row[variant+'_p95_ms'] = sorted(values)[math.ceil(len(values)*.95)-1]
        summaries.append(row)
    controls = []
    for eligible in (0,1,10,100,1000,5183):
        for text in ('cell OR cancer','"T cells"','zznomatchzz'):
            row=dict(eligible=eligible,text=text)
            for variant in ('in','exists'):
                values=[r['ms'] for r in raw['controls'] if r['eligible']==eligible and r['text']==text and r['variant']==variant]
                assert len(values)==3
                row[variant+'_p50_ms']=statistics.median(values)
            controls.append(row)
    return dict(sqlite=raw['sqlite'],timings=summaries,controls=controls)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    raw=json.loads(gzip.decompress((HERE/'recorded/results.json.gz').read_bytes()))
    result=summarize(raw)
    path=HERE/'recorded/summary.json'
    if args.write:
        path.write_text(json.dumps(result,indent=2)+'\n')
    else:
        assert json.loads(path.read_text())==result
        for name,digest in json.loads((HERE/'recorded/hashes.json').read_text()).items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    print(json.dumps(result,indent=2))
