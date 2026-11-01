"""Profile and compare scoped FTS query plans without changing retrieval semantics."""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import platform
import random
import sqlite3
import statistics
import tempfile
import time
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
PREVIOUS=HERE.parent/'2026-10-26-scifact-hybrid'
spec=importlib.util.spec_from_file_location('scifact',PREVIOUS/'evaluate.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def query(guide):
    return guide.read_text().split('<!-- scoped-hybrid-query -->',1)[1].split('```sql\n',1)[1].split('```',1)[0]


def sqls():
    old=query(HERE/'recorded/baseline-guide.md')
    new=query(REPO/'site/guides/hybrid-search.md')
    keyword=s.KEYWORD.replace('rowid in (select id from eligible)',
        'exists (select 1 from eligible where eligible.id = fts_docs.rowid)')
    assert 'exists (select 1 from eligible where eligible.id = fts_docs.rowid)' in new
    return {'keyword':{'baseline':s.KEYWORD,'candidate':keyword},'hybrid':{'baseline':old,'candidate':new}}


def load():
    work=PREVIOUS/'.work'
    manifest=json.loads((PREVIOUS/'recorded/data-manifest.json').read_text())
    for name in ('data.json','documents.npy','queries.npy'):
        assert hashlib.sha256((work/name).read_bytes()).hexdigest()==manifest['files'][name],name
    data=json.loads((work/'data.json').read_text())
    return data,np.load(work/'documents.npy'),np.load(work/'queries.npy')


def execute(db,sql,p):
    start=time.perf_counter_ns();cpu=time.process_time_ns()
    rows=db.execute(sql,p).fetchall()
    return rows,dict(wall_us=(time.perf_counter_ns()-start)/1000,cpu_us=(time.process_time_ns()-cpu)/1000)


def summary(raw):
    rows=[]
    for scenario in ('full','scoped'):
        for method in ('keyword','hybrid'):
            pairs=[p for p in raw['timings'] if p['scenario']==scenario and p['method']==method]
            old=sorted(p['baseline']['wall_us'] for p in pairs);new=sorted(p['candidate']['wall_us'] for p in pairs)
            ratios=[p['baseline']['wall_us']/p['candidate']['wall_us'] for p in pairs]
            rows.append(dict(scenario=scenario,method=method,pairs=len(pairs),
                baseline_p50_us=statistics.median(old),candidate_p50_us=statistics.median(new),
                baseline_p95_us=old[math.ceil(.95*len(old))-1],candidate_p95_us=new[math.ceil(.95*len(new))-1],
                geometric_mean_speedup=math.exp(statistics.mean(map(math.log,ratios)))))
    return rows


def run(work,rounds):
    data,vectors,queries=load();ids=[int(d['_id']) for d in data['documents']]
    archived=json.loads(gzip.decompress((PREVIOUS/'recorded/receipts.json.gz').read_bytes()))
    gold={(r['scenario'],r['query_id'],r['method']):r for r in archived['results']}
    expressions=s.lexical_queries(data['queries']);timed=set(archived['timed_query_ids'])
    library=work/'build/vec0.dylib';variants=sqls()
    raw=dict(quality=[],timings=[],controls=[],rounds=rounds,timed_query_ids=sorted(timed,key=int),plans=[])
    with tempfile.TemporaryDirectory(prefix='fts-comparison-') as directory:
        db=s.connect(Path(directory)/'benchmark.db',library);s.setup(db,data['documents'],vectors)
        for scenario in ('full','scoped'):
            s.apply_scope(db,ids,scenario,set())
            valid={i for i in ids if s.allowed(s.scope(i,scenario,set()))}
            for qi,q in enumerate(data['queries']):
                qid=q['_id'];p=dict(tenant='ours',as_of=100,q_vec=queries[qi].tobytes(),q_text=expressions[qid],branch_limit=50)
                # Full current-build candidate output must reproduce the frozen baseline.
                vector=db.execute(s.VECTOR,p).fetchall()
                assert [list(r) for r in vector]==gold[scenario,qid,'vector']['candidates']
                for method,pair in variants.items():
                    expected=gold[scenario,qid,method]['candidates']
                    candidate=db.execute(pair['candidate'],p).fetchall()
                    assert [list(r) for r in candidate]==expected,(scenario,qid,method)
                    assert {r[0] for r in candidate}<=valid
                    judgments={i:g for i,g in data['qrels'][qid].items() if int(i) in valid}
                    metrics=s.metrics([r[0] for r in candidate[:10]],judgments)
                    assert metrics==gold[scenario,qid,method]['metrics']
                    raw['quality'].append(dict(scenario=scenario,query_id=qid,method=method,rows=candidate,metrics=metrics))
                    if qi==0:
                        for name,sql in pair.items():raw['plans'].append(dict(scenario=scenario,method=method,variant=name,plan=db.execute('explain query plan '+sql,p).fetchall()))
                    if qid not in timed:continue
                    for name,sql in pair.items():
                        warm=db.execute(sql,p).fetchall()
                        assert [list(r) for r in warm]==expected,(scenario,qid,method,name)
                    for repetition in range(rounds):
                        order=['baseline','candidate'];random.Random(42+qi*100+repetition).shuffle(order)
                        result=dict(scenario=scenario,query_id=qid,method=method,round=repetition,order=order)
                        for name in order:
                            rows,measurement=execute(db,pair[name],p)
                            assert [list(r) for r in rows]==expected
                            result[name]=measurement
                        raw['timings'].append(result)
                if qi%25==0:print(f'{scenario}: {qi+1}/300 candidate equivalence checks and sampled timings',flush=True)
        # Small-scope and rare-query controls expose costs hidden by dense averages.
        s.apply_scope(db,ids,'full',set())
        for count in (0,1,10,100,1000,5183):
            db.execute('begin');db.execute("update doc_scope set tenant='other'")
            db.executemany("update doc_scope set tenant='ours' where id=?",[(i,) for i in ids[:count]]);db.execute('commit')
            for qi in (0,1):
                q=data['queries'][qi];p=dict(tenant='ours',as_of=100,q_text=expressions[q['_id']],branch_limit=50)
                expected=None;record=dict(eligible=count,query_id=q['_id'],timings=[])
                for repetition in range(3):
                    pair={};order=['baseline','candidate'];random.Random(count+qi+repetition).shuffle(order)
                    for name in order:
                        rows,measurement=execute(db,variants['keyword'][name],p)
                        if expected is None:expected=rows
                        assert rows==expected
                        assert {r[0] for r in rows}<=set(ids[:count])
                        pair[name]=measurement
                    record['timings'].append(pair)
                raw['controls'].append(record)
        db.close()
    raw['environment']=dict(head=json.loads((work/'build-manifest.json').read_text())['head'],
        extension_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_guide_sha256=hashlib.sha256((HERE/'recorded/baseline-guide.md').read_bytes()).hexdigest(),
        candidate_guide_sha256=hashlib.sha256((REPO/'site/guides/hybrid-search.md').read_bytes()).hexdigest(),
        reference_receipts_sha256=hashlib.sha256((PREVIOUS/'recorded/receipts.json.gz').read_bytes()).hexdigest(),
        sqlite=sqlite3.sqlite_version,python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,
        timing='unprofiled paired sequential queries, shuffled order; one warmup each; same database/library; fixed 30-query sample; CPU unpinned')
    (work/'comparison.json.gz').write_bytes(gzip.compress(json.dumps(raw).encode(),mtime=0))
    (work/'summary.json').write_text(json.dumps(summary(raw),indent=2)+'\n')
    print('Completed',len(raw['quality']),'equivalence points and',len(raw['timings']),'timing pairs',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=HERE/'.work')
    parser.add_argument('--rounds',type=int,default=5)
    args=parser.parse_args()
    if args.rounds<1:parser.error('rounds must be positive')
    run(args.work,args.rounds)
