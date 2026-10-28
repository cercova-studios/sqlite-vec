"""Verify archived results and regenerate relevance and summary metrics offline."""
from pathlib import Path
import ast
import gzip
import hashlib
import json
import math
import random
import evaluate as ev


def main():
    root=Path(__file__).resolve().parent
    for name,digest in json.loads((root/'manifest.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    for path in root.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    raw=json.loads(gzip.decompress((root/'recorded/receipts.json.gz').read_bytes()))
    env=raw['environment']
    assert env['harness_sha256']==hashlib.sha256((root/'evaluate.py').read_bytes()).hexdigest()
    assert env['guide_sha256']==hashlib.sha256((root/'recorded/guide.md').read_bytes()).hexdigest()
    assert env['data_manifest_sha256']==hashlib.sha256((root/'recorded/data-manifest.json').read_bytes()).hexdigest()
    build=json.loads((root/'recorded/build-manifest.json').read_text())
    assert env['extension_sha256']==build['extension_sha256'] and env['build_head']==build['head']
    data=json.loads((root/'recorded/data-manifest.json').read_text())
    assert data['documents']==5183 and data['queries']==300 and data['positive_judgments']==339
    assert data['padding_check'] and data['dimensions']==384
    assert len(raw['document_ids'])==len(set(raw['document_ids']))==5183
    assert len(raw['judgments'])==300 and sum(map(len,raw['judgments'].values()))==339
    assert {g for rels in raw['judgments'].values() for g in rels.values()}=={1}
    assert len(raw['results'])==1800
    expected_timed=set(random.Random(42).sample(sorted(raw['judgments'],key=int),30))
    assert set(raw['timed_query_ids'])==expected_timed
    identities=set();timings=0
    valid_sets={scenario:{i for i in raw['document_ids'] if ev.allowed(ev.scope(i,scenario,set()))} for scenario in ('full','scoped')}
    grouped={}
    for row in raw['results']:
        scenario=row['scenario'];valid=valid_sets[scenario]
        assert set(row['ids'])<=valid
        assert row['ids']==[r[0] for r in row['rows']]
        assert [r[0] for r in row['candidates'][:10]]==row['ids']
        assert len(row['candidates'])==50 if row['method']=='vector' else len(row['candidates'])<=50
        assert {r[0] for r in row['candidates']}<=valid
        judgments={i:g for i,g in raw['judgments'][row['query_id']].items() if int(i) in valid}
        assert ev.metrics(row['ids'],judgments)==row['metrics']
        count=5 if row['query_id'] in expected_timed else 0
        assert len(row['latencies_us'])==len(row['cpu_us'])==count
        assert all(math.isfinite(t) and t>=0 for t in row['latencies_us']+row['cpu_us'])
        timings+=count
        key=(scenario,row['query_id'],row['branch_limit'],row['method'])
        assert row['branch_limit']==50 and key not in identities
        identities.add(key)
        grouped.setdefault(key[:3],{})[row['method']]=row
    for group in grouped.values():
        vr={r[0]:n for n,r in enumerate(group['vector']['candidates'],1)}
        kr={r[0]:n for n,r in enumerate(group['keyword']['candidates'],1)}
        score={i:sum(1/(60+r[i]) for r in (vr,kr) if i in r) for i in vr.keys()|kr.keys()}
        assert group['hybrid']['ids']==sorted(score,key=lambda i:(-score[i],i))[:10]
        for i,s,v,k,distance,source in group['hybrid']['rows']:
            assert math.isclose(s,score[i],abs_tol=1e-12)
            assert (v,k,source)==(vr.get(i),kr.get(i),'scifact:'+str(i))
    assert timings==900
    assert ev.summarize(raw)==json.loads((root/'recorded/summary.json').read_text())
    assert json.loads((root/'recorded/metric-oracle.json').read_text())['passed']
    print('Archive verified: hashes/provenance, 1800 quality points, 900 timed queries, eligibility, fusion and regenerated metrics.')


if __name__=='__main__':
    main()
