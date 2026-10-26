"""Verify recorded results and regenerate metrics without rerunning timings."""
from pathlib import Path
import ast
import gzip
import hashlib
import json
import math
import evaluate as ev


def main():
    root = Path(__file__).resolve().parent
    for name,digest in json.loads((root/'manifest.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    for path in root.glob('*.py'):
        ast.parse(path.read_text(),filename=str(path))
    receipts = json.loads(gzip.decompress((root/'recorded/receipts.json.gz').read_bytes()))
    env = receipts['environment']
    assert env['harness_sha256'] == hashlib.sha256((root/'evaluate.py').read_bytes()).hexdigest()
    assert env['fixture_sha256'] == hashlib.sha256((root/'fixture.json').read_bytes()).hexdigest()
    assert env['guide_sha256'] == hashlib.sha256((root/'recorded/guide.md').read_bytes()).hexdigest()
    build = json.loads((root/'recorded/build-manifest.json').read_text())
    assert env['extension_sha256'] == build['extension_sha256']
    assert len(receipts['results']) == 1944
    assert len(receipts['costs']) == 9 and len(receipts['late_filter_controls']) == 36
    models = {}
    keys = set()
    for row in receipts['results']:
        key = (row['distractors'],row['insertion_seed'],row['phase'])
        if key not in models:
            docs = ev.corpus(key[0],key[1])
            if key[2] != 'baseline':
                docs[ev.FIXTURE['invalidate_id']]['invalidated_at'] = 100
            if key[2] in ('updated','reopened'):
                doc = docs[ev.FIXTURE['update_id']]
                doc.update(ev.FIXTURE['update'])
                doc['version'] += 1
                doc['source_id'] += ':replacement'
            models[key] = docs
        query = next(q for q in ev.FIXTURE['queries'] if q['name'] == row['query'])
        metrics = ev.assess([(i,) for i in row['ids']],models[key],query)
        for name,value in metrics.items():
            assert row[name] == value,(key,name)
        assert not row['ineligible_ids'] and not row['stale_ids']
        assert len(row['latencies_us']) == len(row['cpu_us']) == env['rounds'] == 9
        assert all(math.isfinite(t) and t >= 0 for t in row['latencies_us']+row['cpu_us'])
        identity = (*key,row['query'],row['branch_limit'],row['method'])
        assert identity not in keys
        keys.add(identity)
    updated = {(r['distractors'],r['insertion_seed'],r['query'],r['branch_limit'],r['method']):r['ids']
               for r in receipts['results'] if r['phase'] == 'updated'}
    for row in receipts['results']:
        if row['phase'] == 'reopened':
            assert row['ids'] == updated[row['distractors'],row['insertion_seed'],row['query'],row['branch_limit'],row['method']]
    summary = ev.summarize(receipts)
    assert summary == json.loads((root/'recorded/summary.json').read_text())
    print('Verified hashes, provenance, 1944 quality points, 17496 timed queries and regenerated metrics.')


if __name__ == '__main__':
    main()
