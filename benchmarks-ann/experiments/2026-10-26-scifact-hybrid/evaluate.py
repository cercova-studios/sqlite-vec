"""Compare current vector, SQLite FTS5 and RRF retrieval on BEIR SciFact."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import platform
import random
import sqlite3
import statistics
import time
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
GUIDE=REPO/'site/guides/hybrid-search.md'
HYBRID=GUIDE.read_text().split('<!-- scoped-hybrid-query -->',1)[1].split('```sql\n',1)[1].split('```',1)[0]
ELIGIBLE='''with eligible as materialized (
 select id from doc_scope where tenant=:tenant and invalidated_at is null
 and valid_from<=:as_of and (valid_to is null or valid_to>:as_of)
)'''
VECTOR=ELIGIBLE+''' select rowid,distance from vec_docs
 where embedding match :q_vec and k=:branch_limit
 and rowid in (select id from eligible) order by distance'''
KEYWORD=ELIGIBLE+''' select rowid,rank from fts_docs
 where fts_docs match :q_text and rowid in (select id from eligible)
 order by rank,rowid limit :branch_limit'''


def scope(doc_id, scenario, invalidated):
    bucket=hashlib.sha256(str(doc_id).encode()).digest()[0]%10
    tenant='ours';start=0;end=None;invalid=None
    if scenario!='full':
        if bucket==0:tenant='other'
        elif bucket==1:end=100
        elif bucket==2:start=101
        elif bucket==3:invalid=99
    if scenario=='invalidated' and doc_id in invalidated:invalid=100
    return tenant,start,end,invalid


def allowed(attributes):
    tenant,start,end,invalid=attributes
    return tenant=='ours' and invalid is None and start<=100 and (end is None or 100<end)


def metrics(ids, judgments):
    assert len(ids)==len(set(ids)) and len(ids)<=10
    positives={int(i):grade for i,grade in judgments.items() if grade>0}
    hit=sum(i in positives for i in ids)
    ideal=sum(grade/math.log2(rank+2) for rank,grade in enumerate(sorted(positives.values(),reverse=True)[:10]))
    dcg=sum(positives.get(i,0)/math.log2(rank+2) for rank,i in enumerate(ids))
    return dict(relevant=len(positives),hits=hit,recall_at_10=hit/len(positives) if positives else None,
                ndcg_at_10=dcg/ideal if ideal else None,reciprocal_rank=next((1/(r+1) for r,i in enumerate(ids) if i in positives),0) if positives else None,
                empty=not ids,unjudged_results=sum(i not in positives for i in ids))


def connect(path, extension):
    db=sqlite3.connect(path,isolation_level=None)
    db.enable_load_extension(True);db.load_extension(str(extension));db.enable_load_extension(False)
    return db


def lexical_queries(queries):
    db=sqlite3.connect(':memory:')
    db.executescript("create virtual table tokens using fts5(text); create virtual table terms using fts5vocab(tokens,'instance');")
    result={}
    for query in queries:
        db.execute('delete from tokens');db.execute('insert into tokens values (?)',(query['text'],))
        terms=list(dict.fromkeys(row[0] for row in db.execute('select term from terms order by offset')))
        assert terms
        result[query['_id']]=' OR '.join('"'+term.replace('"','""')+'"' for term in terms)
    db.close()
    return result


def setup(db, documents, vectors):
    db.executescript(f'''create table doc_scope(id integer primary key,tenant text,
     valid_from integer,valid_to integer,invalidated_at integer,source_id text);
     create virtual table vec_docs using vec0(embedding float[{vectors.shape[1]}]);
     create virtual table fts_docs using fts5(body);''')
    t=time.perf_counter_ns();db.execute('begin')
    for doc,vector in zip(documents,vectors):
        i=int(doc['_id'])
        db.execute('insert into doc_scope values (?,?,?,?,?,?)',(i,'ours',0,None,None,'scifact:'+str(i)))
        db.execute('insert into vec_docs(rowid,embedding) values (?,?)',(i,vector.tobytes()))
        db.execute('insert into fts_docs(rowid,body) values (?,?)',(i,doc['title']+' '+doc['text']))
    db.execute('commit')
    return (time.perf_counter_ns()-t)/1e6


def apply_scope(db, ids, scenario, invalidated):
    t=time.perf_counter_ns();db.execute('begin')
    for i in ids:
        attributes=scope(i,scenario,invalidated)
        db.execute('update doc_scope set tenant=?,valid_from=?,valid_to=?,invalidated_at=? where id=?',(*attributes,i))
    db.execute('commit')
    return (time.perf_counter_ns()-t)/1e6


def storage(db):
    return dict(allocated_bytes=db.execute('pragma page_count').fetchone()[0]*db.execute('pragma page_size').fetchone()[0],
                freelist_pages=db.execute('pragma freelist_count').fetchone()[0])


def check_candidates(candidates, fused, distances, valid_ids, limit):
    vector=candidates['vector'];keyword=candidates['keyword']
    assert len(vector)==min(limit,len(valid_ids))
    assert all(row[0] in valid_ids for rows in candidates.values() for row in rows)
    cutoff=sorted(distances[i] for i in valid_ids)[len(vector)-1]
    found={row[0] for row in vector}
    assert {i for i in valid_ids if distances[i]<cutoff-1e-6}<=found
    for i,distance in vector:
        assert distances[i]<=cutoff+1e-6
        assert math.isclose(distance,distances[i],rel_tol=2e-5,abs_tol=2e-5)
    vr={row[0]:rank for rank,row in enumerate(vector,1)}
    kr={row[0]:rank for rank,row in enumerate(keyword,1)}
    scores={i:sum(1/(60+r[i]) for r in (vr,kr) if i in r) for i in vr.keys()|kr.keys()}
    order=sorted(scores,key=lambda i:(-scores[i],i))[:10]
    assert [r[0] for r in fused]==order
    for i,score,v,k,distance,source in fused:
        assert (v,k,source)==(vr.get(i),kr.get(i),'scifact:'+str(i))
        assert math.isclose(score,scores[i],abs_tol=1e-12)
        assert (distance is None)==(i not in vr)
        if distance is not None:assert math.isclose(distance,distances[i],rel_tol=2e-5,abs_tol=2e-5)


def summarize(receipts):
    rows=[]
    for scenario in ('full','scoped'):
        for limit in (50,):
            for method in ('vector','keyword','hybrid'):
                group=[r for r in receipts['results'] if (r['scenario'],r['branch_limit'],r['method'])==(scenario,limit,method)]
                answerable=[r for r in group if r['metrics']['relevant']]
                lat=sorted(t for r in group for t in r['latencies_us'])
                rows.append(dict(scenario=scenario,branch_limit=limit,method=method,queries=len(group),answerable=len(answerable),
                    recall_at_10=statistics.mean(r['metrics']['recall_at_10'] for r in answerable),
                    ndcg_at_10=statistics.mean(r['metrics']['ndcg_at_10'] for r in answerable),
                    mrr_at_10=statistics.mean(r['metrics']['reciprocal_rank'] for r in answerable),
                    empty_rate=statistics.mean(r['metrics']['empty'] for r in group),
                    answerable_empty_rate=statistics.mean(r['metrics']['empty'] for r in answerable),
                    unjudged_fraction=sum(r['metrics']['unjudged_results'] for r in group)/sum(len(r['ids']) for r in group),
                    p50_us=statistics.median(lat),p95_us=lat[math.ceil(.95*len(lat))-1]))
    return rows


def run(work, extension, rounds, output):
    data=json.loads((work/'data.json').read_text());docs=data['documents'];queries=data['queries'];qrels=data['qrels']
    vectors=np.load(work/'documents.npy');query_vectors=np.load(work/'queries.npy')
    assert vectors.shape==(5183,384) and query_vectors.shape==(300,384)
    assert np.isfinite(vectors).all() and np.isfinite(query_vectors).all()
    ids=[int(d['_id']) for d in docs]
    invalidated=set(sorted(i for i in ids if allowed(scope(i,'scoped',set())))[:32])
    words=lexical_queries(queries)
    timed_ids=set(random.Random(42).sample([q['_id'] for q in queries],30))
    # Each run owns a fresh, disposable database. Downloads/arrays remain reusable.
    import tempfile
    receipts=dict(results=[],scenario_costs=[],query_plans=[],invalidated_ids=sorted(invalidated),
                  judgments=qrels,lexical_expressions=words,document_ids=ids,timed_query_ids=sorted(timed_ids,key=int))
    with tempfile.TemporaryDirectory(prefix='scifact-hybrid-') as folder:
        db=connect(Path(folder)/'corpus.db',extension)
        receipts['insert_commit_ms']=setup(db,docs,vectors)
        reference_vectors=vectors.astype(np.float64)
        for scenario in ('full','scoped'):
            cost=apply_scope(db,ids,scenario,invalidated)
            valid={i for i in ids if allowed(scope(i,scenario,invalidated))}
            receipts['scenario_costs'].append(dict(scenario=scenario,eligible_documents=len(valid),metadata_commit_ms=cost,storage=storage(db)))
            for qi,query in enumerate(queries):
                qid=query['_id'];q=query_vectors[qi]
                delta=reference_vectors-q.astype(np.float64)
                reference=np.sqrt(np.einsum('ij,ij->i',delta,delta))
                distances=dict(zip(ids,reference.tolist()))
                judgments={i:g for i,g in qrels[qid].items() if int(i) in valid}
                for limit in (50,):
                    p=dict(tenant='ours',as_of=100,q_vec=q.tobytes(),q_text=words[qid],branch_limit=limit)
                    candidates={m:db.execute(sql,p).fetchall() for m,sql in [('vector',VECTOR),('keyword',KEYWORD)]}
                    fused=db.execute(HYBRID,p).fetchall()
                    check_candidates(candidates,fused,distances,valid,limit)
                    measured={}
                    for method,rows in [('vector',candidates['vector']),('keyword',candidates['keyword']),('hybrid',fused)]:
                        returned=rows[:10];returned_ids=[r[0] for r in returned]
                        assert set(returned_ids)<=valid
                        measured[method]=dict(scenario=scenario,query_id=qid,branch_limit=limit,method=method,
                            ids=returned_ids,rows=returned,candidates=rows,metrics=metrics(returned_ids,judgments),latencies_us=[],cpu_us=[])
                    sqls=dict(vector=VECTOR,keyword=KEYWORD,hybrid=HYBRID)
                    if qi==0:
                        for method,sql in sqls.items():
                            receipts['query_plans'].append(dict(scenario=scenario,branch_limit=limit,method=method,
                                plan=db.execute('explain query plan '+sql,p).fetchall()))
                    if qid in timed_ids:
                        for method,sql in sqls.items():db.execute(sql,p).fetchall()
                    for repetition in range(rounds if qid in timed_ids else 0):
                        order=list(sqls);random.Random(qi*100+repetition).shuffle(order)
                        for method in order:
                            t=time.perf_counter_ns();cpu=time.process_time_ns()
                            result=db.execute(sqls[method],p).fetchall()[:10]
                            measured[method]['latencies_us'].append((time.perf_counter_ns()-t)/1000)
                            measured[method]['cpu_us'].append((time.process_time_ns()-cpu)/1000)
                            assert [r[0] for r in result]==measured[method]['ids']
                    receipts['results'].extend(measured.values())
                if qi%10==0:print(f'{scenario}: {qi+1}/300 quality queries checked; fixed sample timed',flush=True)
            # Reopen checks the actual committed metadata and retrieval without retiming.
            db.close();db=connect(Path(folder)/'corpus.db',extension)
            assert {r[0] for r in db.execute("select id from doc_scope where tenant='ours' and valid_from<=100 and (valid_to is null or valid_to>100) and invalidated_at is null")}==valid
        t=time.perf_counter_ns();db.execute("insert into fts_docs(fts_docs) values ('optimize')")
        receipts['fts_optimize_ms']=(time.perf_counter_ns()-t)/1e6
        receipts['final_storage']=storage(db);db.close()
    build=json.loads((work/'build-manifest.json').read_text())
    assert hashlib.sha256(extension.read_bytes()).hexdigest()==build['extension_sha256']
    receipts['environment']=dict(build_head=build['head'],extension_sha256=build['extension_sha256'],
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),guide_sha256=hashlib.sha256(GUIDE.read_bytes()).hexdigest(),
        data_manifest_sha256=hashlib.sha256((work/'data-manifest.json').read_bytes()).hexdigest(),rounds=rounds,
        python=platform.python_version(),numpy=np.__version__,sqlite=sqlite3.sqlite_version,platform=platform.platform(),
        timing='warm single-client sequential SQL; CPU unpinned; DELETE journal; full branch budget then top 10; embedding time excluded')
    output.mkdir(parents=True,exist_ok=True)
    (output/'receipts.json.gz').write_bytes(gzip.compress(json.dumps(receipts).encode(),mtime=0))
    (output/'summary.json').write_text(json.dumps(summarize(receipts),indent=2)+'\n')
    print('Saved',len(receipts['results']),'quality points',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=HERE/'.work')
    parser.add_argument('--rounds',type=int,default=5)
    args=parser.parse_args()
    if args.rounds<1:parser.error('rounds must be positive')
    run(args.work,args.work/'build/vec0.dylib',args.rounds,args.work/'results')


if __name__=='__main__':
    main()
