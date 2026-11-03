"""Recompute benchmark summaries and reject incomplete or corrupt receipts."""
from pathlib import Path
import argparse, copy, hashlib, json, math
import bench

HERE = Path(__file__).resolve().parent

def validate(result):
    assert result['source_sha256'] == bench.sha((HERE/'bench.py').read_bytes())
    assert result['graphs_sha256'] == bench.sha(json.dumps(bench.graphs()).encode())
    assert result['summary'] == bench.summary(result['runs'])
    assert len(result['runs']) == 10
    seen = set()
    for r in result['runs']:
        key = (r['variant'],r['round'])
        assert key not in seen and 0 <= r['round'] < 5 and r['variant'] in ['before','after']
        seen.add(key)
        assert len(r['points']) == 12
        assert r['work_100000'] == ('passed' if r['variant'] == 'after' else 'graph_witness: work limit exceeded')
        for p in r['points']:
            assert math.isfinite(p['ms']) and p['ms'] >= 0
            assert len(p['path_sha256']) == 64
            edges,pairs = bench.graphs()[p['graph']]
            assert (p['source'],p['target']) in pairs
            hops = bench.oracle(edges,p['source'],p['target'])
            assert p['path_nodes'] == (0 if hops is None else hops+1)
    for variant in ['before','after']:
        for probes in result['payload_budget_probes'][variant].values():
            assert [p['bytes'] for p in probes] == [1<<i for i in range(7,25)]
            assert all(type(p['passed']) is bool for p in probes) and any(p['passed'] for p in probes)
    before,after = result['libraries']['before'],result['libraries']['after']
    assert before.keys() == after.keys()
    for name in before:
        if name not in ['sqlite-vec-graph.c','vec0.dylib']:
            assert before[name] == after[name]
    assert before['sqlite-vec-graph.c'] != after['sqlite-vec-graph.c']
    assert after['sqlite-vec-graph.c'] == bench.sha((bench.REPO/'sqlite-vec-graph.c').read_bytes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    args = parser.parse_args()
    result = json.loads((HERE/'recorded/results.json').read_text())
    validate(result)
    for field,value in [('path_nodes',-1),('path_sha256','x'*64)]:
        bad = copy.deepcopy(result)
        bad['runs'][0]['points'][0][field] = value
        try:
            validate(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corruption accepted: '+field)
    path = HERE/'recorded/hashes.json'
    if args.write:
        files = sorted(p for p in HERE.rglob('*') if p.is_file() and '.work' not in p.parts and '__pycache__' not in p.parts and p != path)
        path.write_text(json.dumps({str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    else:
        for name,digest in json.loads(path.read_text()).items():
            assert bench.sha((HERE/name).read_bytes()) == digest,name
    print('Verified 120 timed witnesses, budget probes, source hashes and corruption rejection')

if __name__ == '__main__':
    main()
