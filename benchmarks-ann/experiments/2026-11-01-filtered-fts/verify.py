"""Verify the frozen comparison and regenerate its timing summary offline."""
from pathlib import Path
import ast
import gzip
import hashlib
import json
import math
import benchmark as b


def verify(root):
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest
    for name,digest in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    for path in root.glob('*.py'):
        ast.parse(path.read_text(),filename=str(path))
    raw=json.loads(gzip.decompress((root/'recorded/comparison.json.gz').read_bytes()))
    env=raw['environment']
    for field,name in [('harness_sha256','benchmark.py'),('baseline_guide_sha256','recorded/baseline-guide.md'),
                       ('candidate_guide_sha256','recorded/candidate-guide.md')]:
        assert env[field]==hashlib.sha256((root/name).read_bytes()).hexdigest(),field
    baseline=b.query(root/'recorded/baseline-guide.md')
    candidate=b.query(root/'recorded/candidate-guide.md')
    assert baseline==b.query(b.PREVIOUS/'recorded/guide.md')
    assert baseline.replace('where fts_docs match :q_text and rowid in (select id from eligible)',
        'where fts_docs match :q_text\n    and exists (select 1 from eligible where eligible.id = fts_docs.rowid)')==candidate
    reference=b.PREVIOUS/'recorded/receipts.json.gz'
    assert env['reference_receipts_sha256']==hashlib.sha256(reference.read_bytes()).hexdigest()
    old=json.loads(gzip.decompress(reference.read_bytes()))
    gold={(r['scenario'],r['query_id'],r['method']):r for r in old['results']}
    build=json.loads((root/'recorded/build-manifest.json').read_text())
    assert env['head']==build['head'] and env['extension_sha256']==build['extension_sha256']
    prior_build=json.loads((b.PREVIOUS/'recorded/build-manifest.json').read_text())
    assert build['source_hashes']==prior_build['source_hashes']
    assert raw['rounds']==5 and set(raw['timed_query_ids'])==set(old['timed_query_ids'])
    identities=set()
    for row in raw['quality']:
        key=(row['scenario'],row['query_id'],row['method'])
        assert key not in identities and row['method'] in ('keyword','hybrid')
        identities.add(key)
        assert row['rows']==gold[key]['candidates']
        assert row['metrics']==gold[key]['metrics']
        valid={i for i in old['document_ids'] if b.s.allowed(b.s.scope(i,row['scenario'],set()))}
        assert {r[0] for r in row['rows']}<=valid
        judgments={i:g for i,g in old['judgments'][row['query_id']].items() if int(i) in valid}
        assert b.s.metrics([r[0] for r in row['rows'][:10]],judgments)==row['metrics']
    assert len(identities)==1200
    pairs=set()
    for pair in raw['timings']:
        key=(pair['scenario'],pair['query_id'],pair['method'],pair['round'])
        assert key not in pairs and pair['query_id'] in raw['timed_query_ids'] and 0<=pair['round']<5
        assert key[:3] in identities and sorted(pair['order'])==['baseline','candidate']
        pairs.add(key)
        for variant in ('baseline','candidate'):
            assert all(math.isfinite(pair[variant][metric]) and pair[variant][metric]>0 for metric in ('wall_us','cpu_us'))
    assert len(pairs)==600
    assert b.summary(raw)==json.loads((root/'recorded/summary.json').read_text())
    assert len(raw['controls'])==12
    assert {(c['eligible'],c['query_id']) for c in raw['controls']}=={
        (n,q) for n in (0,1,10,100,1000,5183) for q in ('1','3')}
    assert all(len(c['timings'])==3 for c in raw['controls'])
    profile=json.loads((root/'recorded/diagnostic.json').read_text())['variants']
    assert profile['baseline']['rows']==profile['exists']['rows']==profile['unary']['rows']
    alternatives=json.loads((root/'recorded/alternatives.json').read_text())
    assert all(a['rows']==alternatives[0]['rows'] for a in alternatives)
    print('Archive verified: hashes, 1200 exact quality points, 600 timing pairs, 12 selectivity controls and regenerated summary.')


if __name__=='__main__':
    verify(Path(__file__).resolve().parent)
