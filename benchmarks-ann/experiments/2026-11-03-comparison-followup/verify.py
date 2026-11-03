"""Verify frozen receipts and regenerate summaries without engine dependencies."""
from pathlib import Path
from collections import defaultdict
import argparse, copy, gzip, hashlib, importlib.util, json, math, statistics

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / '2026-11-03-engine-comparison'
REC = HERE / 'recorded'
spec = importlib.util.spec_from_file_location('vector_verifier', PREVIOUS / 'verify.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(name):
    path = REC / name
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_text())

def validate_vector(result, oracle):
    modes = [('all', 'ann')] if result['engine'] == 'libsql' else []
    modes += [(case, 'scan') for case in oracle['truth']]
    assert len(result['quality']) == len(modes) * 100
    assert len(result['timings']) == len(modes) * 300
    seen = set()
    for p in result['quality']:
        key = (p['case'], p['mode'], p['query'])
        assert key not in seen and key[:2] in modes and 0 <= p['query'] < 100
        seen.add(key)
        case = oracle['truth'][p['case']]
        truth = case['queries'][p['query']]
        ids = [r[0] for r in p['rows']]
        assert len(ids) == len(set(ids)) and len(ids) <= len(truth['ids'])
        assert all(type(i) is int and 0 <= i < len(oracle['bucket']) and oracle[case['field']][i] < case['threshold'] for i in ids)
        assert all(math.isfinite(r[1]) for r in p['rows'])
        assert previous.recall(ids, truth)[1] == p['recall']
        if result['engine'] == 'libsql':
            assert p['distance_accurate']
            if p['mode'] == 'scan':
                assert p['recall'] == 1
    seen = set()
    for p in result['timings']:
        key = (p['case'], p['mode'], p['round'], p['query'])
        assert key not in seen and key[:2] in modes
        assert 0 <= p['round'] < 3 and 0 <= p['query'] < 100
        assert math.isfinite(p['ms']) and p['ms'] >= 0 and 0 <= p['recall'] <= 1
        seen.add(key)
        if result['engine'] == 'libsql':
            assert p['distance_accurate']
            if p['mode'] == 'scan':
                assert p['recall'] == 1
    return [dict(engine=result['engine'], dataset=result['dataset'], case=case, mode=mode,
                 p50_ms=statistics.median(p['ms'] for p in result['timings'] if (p['case'], p['mode']) == (case, mode)),
                 recall=statistics.mean(p['recall'] for p in result['timings'] if (p['case'], p['mode']) == (case, mode)))
            for case, mode in modes]

def filter_summary():
    raw = load('filter-profile.json')
    assert len(raw['base']) == len(raw['instrumented']) == 60
    assert [p['rows'] for p in raw['base']] == [p['rows'] for p in raw['instrumented']]
    events = {}
    active = None
    for line in (REC / 'instrumented.trace').read_text().splitlines():
        fields = line.split()
        if fields[0] == 'POINT':
            active = (fields[1], int(fields[2]))
            assert active not in events
            events[active] = dict(collector_ns=0)
        elif active and fields[0] == 'COLLECT':
            events[active]['collector_ns'] += int(fields[1])
        elif active and fields[0] == 'HNSW':
            events[active].update(hnsw_ns=int(fields[1]), graph_distance_calls=int(fields[2]), membership_checks=int(fields[3]))
            active = None  # Ignore warmups for the next case.
    assert len(events) == 60
    summary = {}
    for case in ['all', 'random10', 'random1']:
        points = [events[case, i] for i in range(20)]
        base = statistics.median(p['ns'] for p in raw['base'] if p['case'] == case) / 1e6
        inst = statistics.median(p['ns'] for p in raw['instrumented'] if p['case'] == case) / 1e6
        summary[case] = dict(baseline_ms=base, instrumented_ms=inst, instrumentation_overhead_fraction=inst/base-1,
                             collector_ms=statistics.median(p['collector_ns'] for p in points)/1e6,
                             hnsw_ms=statistics.median(p['hnsw_ns'] for p in points)/1e6,
                             graph_search_distance_calls=statistics.median(p['graph_distance_calls'] for p in points),
                             membership_checks=statistics.median(p['membership_checks'] for p in points))
    return summary

def summarize():
    runtime = load('runtime.json')
    assert runtime['prior_inputs_sha256'] == sha(PREVIOUS / 'recorded/inputs.json')
    assert runtime['prior_oracle_sha256'] == sha(PREVIOUS / 'recorded/oracle.json.gz')
    for name, digest in runtime['fixtures'].items():
        assert sha(HERE.parents[2] / name) == digest, name
    oracle = json.loads(gzip.decompress((PREVIOUS / 'recorded/oracle.json.gz').read_bytes()))
    vectors = []
    for engine in ['libsql', 'turso']:
        for dataset in ['scifact', 'sift50k']:
            result = load(f'{engine}-{dataset}.json.gz')
            assert result['engine'] == engine and result['dataset'] == dataset
            assert result['source_hash'] == sha(HERE / 'sql_engines.py')
            vectors.extend(validate_vector(result, oracle[dataset]))
    controls = load('sql-controls.json')
    assert len(controls['measurements']) == 300
    assert controls['summary_ms'] == {name: statistics.median(p['ms'] for p in controls['measurements'] if p['name'] == name) for name in controls['plans']}
    graphs = []
    for name in ['graph-sqlite.json', 'graph-ladybug.json', 'graph-sqlite-deep-probe.json']:
        g = load(name)
        assert g['source_hash'] == sha(HERE / 'graph_compare.py') and len(g['g0']) == 17
        groups = defaultdict(list)
        for p in g['timings']:
            assert math.isfinite(p['ms']) and p['ms'] >= 0
            if p['graph'] == 'chain':
                assert p['path_nodes'] == p['target'] - p['source'] + 1
            groups[p['mode'], p['source'] if p['graph'] == 'chain' else None].append(p['ms'])
        graphs += [dict(receipt=name, mode=mode, source=source, count=len(ms), p50_ms=statistics.median(ms)) for (mode, source), ms in groups.items()]
        if g['engine'] == 'sqlite':
            assert 'work limit exceeded' in g['work_failure']
    failed = load('graph-ladybug-deep-probe.json')
    assert failed['source_hash'] == sha(HERE / 'graph_compare.py') and failed['status'] == 'failed'
    assert 'buffer pool is full' in failed['error']
    for p in load('transaction-probes.json'):
        assert p['passed'] and p['stages'] == ['own writes', 'savepoint', 'full rollback', 'commit/reopen']
    flat = load('flat-reads.json')
    truth = oracle['sift50k']['truth']['random1']['queries'][0]
    assert previous.recall([r[0] for r in flat['rows']], truth)[1] == 1
    assert flat['eligible_count'] == sum(i < 10 for i in oracle['sift50k']['bucket'])
    assert flat['eligible_vector_bytes'] == flat['eligible_count'] * 128 * 4
    assert flat['vector_blob_bytes'] == flat['vector_blob_reads'] * 1024 * 128 * 4
    assert sum(flat['read_sizes']) == flat['vector_blob_bytes'] and len(flat['read_sizes']) == flat['vector_blob_reads']
    assert flat['read_amplification'] == flat['vector_blob_bytes'] / flat['eligible_vector_bytes']
    return dict(vectors=vectors, filters=filter_summary(), sql_controls_ms=controls['summary_ms'], graphs=graphs, flat=flat)

def self_test():
    previous.self_test()
    oracle = json.loads(gzip.decompress((PREVIOUS / 'recorded/oracle.json.gz').read_bytes()))['sift50k']
    original = load('libsql-sift50k.json.gz')
    for kind in ['id', 'recall']:
        bad = copy.deepcopy(original)
        if kind == 'id':
            bad['quality'][0]['rows'][0][0] = -1
        else:
            bad['quality'][0]['recall'] = -1
        try:
            validate_vector(bad, oracle)
        except AssertionError:
            pass
        else:
            raise AssertionError('corruption accepted: ' + kind)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    self_test()
    result = summarize()
    path = REC / 'summary.json'
    if args.write:
        path.write_text(json.dumps(result, indent=2) + '\n')
        paths = sorted(p for p in HERE.rglob('*') if p.is_file() and '.work' not in p.parts and '__pycache__' not in p.parts and p.name != 'hashes.json')
        (REC / 'hashes.json').write_text(json.dumps({str(p.relative_to(HERE)): sha(p) for p in paths}, indent=2) + '\n')
    else:
        assert result == load('summary.json')
        for name, digest in load('hashes.json').items():
            assert sha(HERE / name) == digest, name
    print('Verified 2200 quality points, 6600 timed calls, diagnostic summaries and corruption rejection')

if __name__ == '__main__':
    main()
