"""Evaluate the existing scoped recipe with independent synthetic judgments."""
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import platform
import random
import sqlite3
import statistics
import struct
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIXTURE = json.loads((HERE / 'fixture.json').read_text())
GUIDE = REPO / 'site/guides/hybrid-search.md'
ELIGIBLE = '''with eligible as materialized (
 select id from doc_scope where tenant=:tenant and invalidated_at is null
 and valid_from<=:as_of and (valid_to is null or valid_to>:as_of)
)'''
VECTOR = ELIGIBLE + ''' select rowid, distance from vec_docs
 where embedding match :q_vec and k=:branch_limit
 and rowid in (select id from eligible) order by distance'''
KEYWORD = ELIGIBLE + ''' select rowid, rank from fts_docs
 where fts_docs match :q_text and rowid in (select id from eligible)
 order by rank,rowid limit :branch_limit'''
HYBRID = GUIDE.read_text().split('<!-- scoped-hybrid-query -->', 1)[1].split('```sql\n', 1)[1].split('```', 1)[0]
LATE = '''with knn as materialized (
 select rowid from vec_docs where embedding match :q_vec and k=:branch_limit
), fts as materialized (
 select rowid from fts_docs where fts_docs match :q_text order by rank,rowid limit :branch_limit
), candidates as (select rowid from knn union select rowid from fts)
select rowid from candidates join doc_scope s on s.id=rowid
where s.tenant=:tenant and s.invalidated_at is null and s.valid_from<=:as_of
and (s.valid_to is null or s.valid_to>:as_of)'''


def blob(vector):
    return struct.pack('4f', *vector)


def canonical(vector):
    return list(struct.unpack('4f', blob(vector)))


def eligible(doc, query):
    return (doc['tenant'] == query['tenant'] and doc['invalidated_at'] is None
            and doc['valid_from'] <= query['as_of']
            and (doc['valid_to'] is None or query['as_of'] < doc['valid_to']))


def expected(documents, query):
    return {d['id'] for d in documents.values() if eligible(d, query) and query['topic'] in d['relevant_to']}


def corpus(count, seed):
    documents = copy.deepcopy(FIXTURE['documents'])
    rng = random.Random(42)
    for i in range(count):
        topic = rng.randrange(4)
        q = [0.0]*4
        q[topic] = 1.0
        # Most distractors are closer than any relevant record but ineligible.
        kind = i % 10
        doc = dict(id=100+i, tenant='ours', valid_from=0, valid_to=None,
                   invalidated_at=None, source_id=f'distractor:{i}', version=1,
                   body=['deploy', 'rotate', 'refund', 'backup'][topic],
                   vector=q, relevant_to=[])
        if kind < 4:
            doc['tenant'] = 'other'
        elif kind < 6:
            doc['valid_to'] = 100
        elif kind < 8:
            doc['valid_from'] = 101
        elif kind == 8:
            doc['invalidated_at'] = 99
        else:
            doc['vector'] = [rng.uniform(-2, 2) for _ in range(4)]
            doc['body'] += ' cafeteria menu announcement unrelated placeholder'
        documents.append(doc)
    random.Random(seed).shuffle(documents)
    return {d['id']: d for d in documents}


def connect(path, extension):
    db = sqlite3.connect(path, isolation_level=None)
    db.enable_load_extension(True)
    db.load_extension(str(extension))
    db.enable_load_extension(False)
    return db


def setup(db, documents):
    db.executescript('''create table doc_scope(id integer primary key,tenant text,
      valid_from integer,valid_to integer,invalidated_at integer,source_id text);
      create virtual table vec_docs using vec0(embedding float[4]);
      create virtual table fts_docs using fts5(body);''')
    start = time.perf_counter_ns()
    db.execute('begin')
    for d in documents.values():
        db.execute('insert into doc_scope values (?,?,?,?,?,?)', tuple(d[k] for k in ('id','tenant','valid_from','valid_to','invalidated_at','source_id')))
        db.execute('insert into vec_docs(rowid,embedding) values (?,?)', (d['id'], blob(d['vector'])))
        db.execute('insert into fts_docs(rowid,body) values (?,?)', (d['id'], d['body']))
    db.execute('commit')
    return (time.perf_counter_ns()-start)/1e6


def params(query, limit):
    return dict(query, q_vec=blob(query['q_vec']), branch_limit=limit)


def retrieve(db, method, query, limit):
    sql = {'vector':VECTOR, 'keyword':KEYWORD, 'hybrid':HYBRID}[method]
    rows = db.execute(sql, params(query, limit)).fetchall()
    return rows[:10]


def assess(rows, documents, query):
    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids)) and len(ids) <= 10
    bad = [i for i in ids if not eligible(documents[i], query)]
    stale = [i for i in ids if documents[i]['invalidated_at'] is not None
             or (documents[i]['valid_to'] is not None and documents[i]['valid_to'] <= query['as_of'])]
    want = expected(documents, query)
    return dict(ids=ids, relevant_ids=sorted(want), hits=len(set(ids)&want),
                recall_at_10=len(set(ids)&want)/len(want) if want else None,
                ineligible_ids=bad, stale_ids=stale, empty=not ids,
                eligible_documents=sum(eligible(d, query) for d in documents.values()))


def check_vector(db, documents, query, limit):
    rows = db.execute(VECTOR, params(query, limit)).fetchall()
    distances = {d['id']:math.dist(canonical(d['vector']), canonical(query['q_vec']))
                 for d in documents.values() if eligible(d, query)}
    assert len(rows) == min(limit, len(distances))
    if rows:
        cutoff = sorted(distances.values())[len(rows)-1]
        returned = {r[0] for r in rows}
        assert {i for i,v in distances.items() if v < cutoff-1e-6} <= returned
        for i, distance in rows:
            assert distances[i] <= cutoff+1e-6
            assert math.isclose(distance, distances[i], abs_tol=1e-5, rel_tol=1e-5)


def check_fusion(db, documents, query, limit):
    p = params(query, limit)
    vectors = db.execute(VECTOR, p).fetchall()
    keywords = db.execute(KEYWORD, p).fetchall()
    vr = {r[0]:n for n,r in enumerate(vectors,1)}
    kr = {r[0]:n for n,r in enumerate(keywords,1)}
    score = {i:sum(1/(60+ranks[i]) for ranks in (vr,kr) if i in ranks) for i in vr.keys()|kr.keys()}
    order = sorted(score, key=lambda i:(-score[i],i))[:10]
    rows = retrieve(db, 'hybrid', query, limit)
    assert [r[0] for r in rows] == order
    for i,s,v,k,distance,source in rows:
        assert math.isclose(s,score[i],abs_tol=1e-12)
        assert (v,k,source) == (vr.get(i),kr.get(i),documents[i]['source_id'])
        assert (distance is None) == (i not in vr)


def mutate(db, documents):
    doc = documents[FIXTURE['update_id']]
    doc.update(copy.deepcopy(FIXTURE['update']))
    doc['version'] += 1
    doc['source_id'] += ':replacement'
    db.execute('update doc_scope set source_id=? where id=?',(doc['source_id'],doc['id']))
    db.execute('update vec_docs set embedding=? where rowid=?',(blob(doc['vector']),doc['id']))
    db.execute('update fts_docs set body=? where rowid=?',(doc['body'],doc['id']))


def storage(db):
    return dict(allocated_bytes=db.execute('pragma page_count').fetchone()[0]*db.execute('pragma page_size').fetchone()[0],
                freelist_pages=db.execute('pragma freelist_count').fetchone()[0],
                live_documents=db.execute('select count(*) from doc_scope').fetchone()[0])


def run(extension, sizes, seeds, rounds):
    results = []; costs = []; controls = []
    with tempfile.TemporaryDirectory(prefix='scoped-hybrid-') as temporary:
        for count in sizes:
            for seed in seeds:
                path = Path(temporary)/f'{count}-{seed}.db'
                documents = corpus(count, seed)
                db = connect(path, extension)
                insert_ms = setup(db, documents)
                base = dict(distractors=count, insertion_seed=seed)
                cost = dict(base, insert_commit_ms=insert_ms, initial_storage=storage(db), mutations={})
                for phase in ('baseline','invalidated','updated','reopened'):
                    if phase == 'invalidated':
                        t = time.perf_counter_ns();db.execute('begin')
                        db.execute('update doc_scope set invalidated_at=100 where id=?',(FIXTURE['invalidate_id'],))
                        db.execute('commit');documents[FIXTURE['invalidate_id']]['invalidated_at']=100
                        cost['mutations']['invalidate_commit_ms']=(time.perf_counter_ns()-t)/1e6
                    elif phase == 'updated':
                        t=time.perf_counter_ns();db.execute('begin');mutate(db,documents);db.execute('commit')
                        cost['mutations']['update_commit_ms']=(time.perf_counter_ns()-t)/1e6
                    elif phase == 'reopened':
                        db.close();db=connect(path,extension)
                    cost[phase+'_storage']=storage(db)
                    for query in FIXTURE['queries']:
                        for limit in (10,50):
                            check_vector(db,documents,query,limit)
                            check_fusion(db,documents,query,limit)
                            measured = {}
                            for method in ('vector','keyword','hybrid'):
                                rows=retrieve(db,method,query,limit)
                                metrics=assess(rows,documents,query)
                                assert not metrics['ineligible_ids'] and not metrics['stale_ids'], metrics
                                measured[method]=dict(base,phase=phase,query=query['name'],branch_limit=limit,method=method,**metrics,
                                                      latencies_us=[],cpu_us=[])
                            # Alternate method order each round, after two warmups per method.
                            for method in measured:
                                for _ in range(2):retrieve(db,method,query,limit)
                            for repetition in range(rounds):
                                order=list(measured);random.Random(seed*100+repetition).shuffle(order)
                                for method in order:
                                    t=time.perf_counter_ns();cpu=time.process_time_ns()
                                    rows=retrieve(db,method,query,limit)
                                    elapsed=(time.perf_counter_ns()-t)/1000
                                    assert [r[0] for r in rows] == measured[method]['ids']
                                    measured[method]['latencies_us'].append(elapsed)
                                    measured[method]['cpu_us'].append((time.process_time_ns()-cpu)/1000)
                            results.extend(measured.values())
                    query=FIXTURE['queries'][0]
                    late=db.execute(LATE,params(query,10)).fetchall()
                    controls.append(dict(base,phase=phase,late_filtered_ids=[r[0] for r in late],
                                         late_relevant_hits=len({r[0] for r in late}&expected(documents,query))))
                cost['final_storage']=storage(db)
                t=time.perf_counter_ns();db.execute("insert into fts_docs(fts_docs) values ('optimize')")
                cost['fts_optimize_ms']=(time.perf_counter_ns()-t)/1e6
                cost['after_fts_optimize_storage']=storage(db)
                cost['database_file_bytes']=path.stat().st_size
                costs.append(cost);db.close()
                print(f'Completed {count} distractors, seed {seed}',flush=True)
    return dict(results=results,costs=costs,late_filter_controls=controls)


def summarize(receipts):
    rows=[]
    for count in sorted({r['distractors'] for r in receipts['results']}):
        for phase in ('baseline','invalidated','updated','reopened'):
            for limit in (10,50):
                for method in ('vector','keyword','hybrid'):
                    selected=[r for r in receipts['results'] if (r['distractors'],r['phase'],r['branch_limit'],r['method'])==(count,phase,limit,method)]
                    recalls=[r['recall_at_10'] for r in selected if r['recall_at_10'] is not None]
                    lat=sorted(v for r in selected for v in r['latencies_us'])
                    rows.append(dict(distractors=count,phase=phase,branch_limit=limit,method=method,
                                     queries=len(selected),macro_recall_at_10=statistics.mean(recalls),
                                     ineligible_results=sum(len(r['ineligible_ids']) for r in selected),
                                     stale_results=sum(len(r['stale_ids']) for r in selected),
                                     empty_rate=statistics.mean(r['empty'] for r in selected),
                                     answerable_empty_rate=statistics.mean(r['empty'] for r in selected if r['relevant_ids']),
                                     p50_us=statistics.median(lat),p95_us=lat[math.ceil(.95*len(lat))-1]))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension',type=Path,default=REPO/'dist/vec0.dylib')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sizes',type=int,nargs='+',default=[0,1000,10000])
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--rounds',type=int,default=9)
    args=parser.parse_args()
    if args.rounds<1 or not args.sizes or min(args.sizes)<0:parser.error('rounds must be positive; sizes must be nonnegative')
    args.output.mkdir(parents=True,exist_ok=True)
    receipts=run(args.extension.resolve(),args.sizes,args.seeds,args.rounds)
    receipts['environment']=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        python=platform.python_version(),sqlite=sqlite3.sqlite_version,platform=platform.platform(),
        extension_sha256=hashlib.sha256(args.extension.read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256((HERE/'fixture.json').read_bytes()).hexdigest(),
        guide_sha256=hashlib.sha256(GUIDE.read_bytes()).hexdigest(),rounds=args.rounds,
        timing='warm sequential single-client SQL; disk-backed SQLite DELETE journal; cache defaults; method order shuffled; unpinned CPU')
    (args.output/'receipts.json.gz').write_bytes(gzip.compress(json.dumps(receipts).encode(),mtime=0))
    (args.output/'summary.json').write_text(json.dumps(summarize(receipts),indent=2)+'\n')

if __name__=='__main__':
    main()
