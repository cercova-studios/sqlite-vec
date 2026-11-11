"""Fresh-process SciFact relevance/scoring and live-encoder end-to-end workers."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import sqlite3
import time

import numpy as np

from build_baselines import EXTENSION, EXTENSION_SHA
from freeze import HERE, OLD, digest
from host_scorer import decode, pair_work, rerank, score_pair
from metrics import metrics

METHODS = ["bm25", "dense", "rrf"]


def connect():
    assert digest(EXTENSION) == EXTENSION_SHA
    db = sqlite3.connect((HERE/".work/corpus.db").as_uri()+"?mode=ro", uri=True, isolation_level=None)
    db.enable_load_extension(True)
    try:
        db.load_extension(str(EXTENSION))
    finally:
        db.enable_load_extension(False)
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA cache_size=-2048")
    db.execute("PRAGMA mmap_size=0")
    return db


def expressions(queries):
    db = sqlite3.connect(":memory:")
    db.executescript("CREATE VIRTUAL TABLE tokens USING fts5(body); CREATE VIRTUAL TABLE words USING fts5vocab(tokens,'instance');")
    words = {}
    for q in queries:
        db.execute("DELETE FROM tokens")
        db.execute("INSERT INTO tokens VALUES(?)", (q["text"],))
        terms = list(dict.fromkeys(r[0] for r in db.execute("SELECT term FROM words ORDER BY offset")))
        assert terms
        words[q["_id"]] = " OR ".join('"'+term.replace('"','""')+'"' for term in terms)
    db.close()
    return words


def first_stage(db, method, text, vector):
    count = 50 if method == "rrf" else 100
    dense = keyword = []
    if method in ["dense", "rrf"]:
        dense = list(db.execute("SELECT rowid,vec_distance_l2(embedding,?) distance FROM vec_docs ORDER BY distance,rowid LIMIT ?",(vector.astype(np.float32).tobytes(),count)))
    if method in ["bm25", "rrf"]:
        keyword = list(db.execute("SELECT rowid,rank FROM fts_docs WHERE fts_docs MATCH ? ORDER BY rank,rowid LIMIT ?",(text,count)))
    if method == "dense":
        return [r[0] for r in dense]
    if method == "bm25":
        return [r[0] for r in keyword]
    scores = {}
    for branch in [dense,keyword]:
        for rank,(key,_) in enumerate(branch,1):
            scores[key] = scores.get(key,0.0)+1/(60+rank)
    return sorted(scores, key=lambda key:(-scores[key],key))


def operation(db, qid, text, expression, vector, method, profile, late=True, query_override=None):
    db.execute("BEGIN")
    try:
        t = time.perf_counter_ns()
        candidates = first_stage(db,method,expression,vector)
        retrieval_ms = (time.perf_counter_ns()-t)/1e6
        result = None
        if late:
            t = time.perf_counter_ns()
            result = rerank(db,qid,text,candidates,profile,query_override=query_override)
            rerank_ms = (time.perf_counter_ns()-t)/1e6
            ids = [row[0] for row in result["scores"][:10]]
        else:
            rerank_ms = 0.0
            ids = candidates[:10]
    finally:
        db.execute("ROLLBACK")
    return dict(ids=ids,candidates=candidates,retrieval_ms=retrieval_ms,rerank_ms=rerank_ms,scoring=result)


def oracle_check(db, inputs, profile, words, dense):
    from oracle import maxsim, work_required
    checked=[]
    for qid in inputs["protocol"]["tuning_ids"][:3]:
        q = inputs["queries"][qid]
        db.execute("BEGIN")
        try:
            candidates=first_stage(db,"rrf",words[qid],dense[qid])
            qblob=db.execute("SELECT a.vectors FROM li_refs r JOIN li_artifacts a ON a.sha256=r.artifact_sha256 WHERE r.role='query' AND r.item_id=?",(int(qid),)).fetchone()[0]
            query=decode(qblob,128,128,1048576)
            for key in candidates[:2]:
                dblob=db.execute("SELECT a.vectors FROM li_refs r JOIN li_artifacts a ON a.sha256=r.artifact_sha256 WHERE r.role='document' AND r.item_id=?",(key,)).fetchone()[0]
                document=decode(dblob,128,4096,8388608)
                expected=maxsim(qblob,dblob,128,134217728)
                observed=score_pair(query,document)
                assert abs(observed-expected.score)<=1e-6+1e-6*abs(expected.score)
                assert expected.work_used==work_required(qblob,dblob,128)==pair_work(len(query),len(document),128)
                checked.append(dict(query=qid,document=key,oracle=expected.score,host=observed,work=expected.work_used))
        finally:
            db.execute("ROLLBACK")
    # Verify native distance units/candidates independently using direct float64 norms.
    source=np.load(OLD/"documents.npy").astype(np.float64)
    doc_ids=np.array([r[0] for r in db.execute("SELECT id FROM sources ORDER BY id")])
    for qid in inputs["protocol"]["tuning_ids"][:3]:
        distances=np.sqrt(np.square(source-dense[qid].astype(np.float64)).sum(axis=1))
        found=list(db.execute("SELECT rowid,vec_distance_l2(embedding,?) d FROM vec_docs ORDER BY d,rowid LIMIT 100",(dense[qid].tobytes(),)))
        lookup=dict(zip(doc_ids,distances))
        assert all(abs(value-lookup[key])<2e-5 for key,value in found)
        cutoff=max(lookup[key] for key,_ in found)
        assert {int(doc_ids[i]) for i,d in enumerate(distances) if d<cutoff-2e-5}<={key for key,_ in found}
    return checked


def load():
    protocol=json.loads((HERE/"recorded/protocol.json").read_text())
    manifest=json.loads((HERE/".work/embedding-manifest.json").read_text())
    assert manifest["protocol_sha256"]==digest(HERE/"recorded/protocol.json")
    selected=json.loads((HERE/".work/selected.json").read_text())
    ids=protocol["tuning_ids"]+protocol["evaluation_ids"]
    query_vectors=np.load(HERE/".work/dense-queries.npy")
    assert query_vectors.shape==(200,384)
    return dict(protocol=protocol,manifest=manifest,queries={q["_id"]:q for q in selected["queries"]},qrels=selected["qrels"]),dict(zip(ids,query_vectors))


def score_worker(output):
    inputs,dense=load();protocol=inputs["protocol"];profile=inputs["manifest"]["profile_id"]
    db=connect();words=expressions(list(inputs["queries"].values()))
    diagnostic_qid=protocol["tuning_ids"][0]
    diagnostic=operation(db,diagnostic_qid,inputs["queries"][diagnostic_qid]["text"],words[diagnostic_qid],dense[diagnostic_qid],"rrf",profile)
    checks=oracle_check(db,inputs,profile,words,dense)
    records=[]
    for split in ["tuning","evaluation"]:
        for offset,qid in enumerate(protocol[split+"_ids"]):
            for method in METHODS:
                result=operation(db,qid,inputs["queries"][qid]["text"],words[qid],dense[qid],method,profile)
                candidate_ids=result["candidates"]
                records.append(dict(query=qid,split=split,method=method,
                                    baseline_ids=candidate_ids[:10],baseline=metrics(candidate_ids[:10],inputs["qrels"][qid],candidate_ids),
                                    late_ids=result["ids"],late=metrics(result["ids"],inputs["qrels"][qid],candidate_ids),
                                    **{k:result[k] for k in ["candidates","scoring","retrieval_ms","rerank_ms"]}))
            if offset%25==0: print(split,offset+1,flush=True)
        if split=="tuning":
            averages={method:sum(r["baseline"]["ndcg10"] for r in records if r["split"]=="tuning" and r["method"]==method)/50 for method in METHODS}
            selection=dict(baseline=max(METHODS,key=lambda method:averages[method]),tuning_ndcg10=averages,primary_late="rrf_late")
            target=HERE/".work/selection.json"
            with target.open("x") as stream:json.dump(selection,stream,indent=2)
    times=[]
    for qid in protocol["timing_ids"][:3]:
        operation(db,qid,inputs["queries"][qid]["text"],words[qid],dense[qid],"rrf",profile)
    for repeat in range(3):
        ids=protocol["timing_ids"][:];random.Random(991+repeat).shuffle(ids)
        for qid in ids:
            methods=METHODS[:];random.Random(int(qid)+repeat).shuffle(methods)
            for method in methods:
                result=operation(db,qid,inputs["queries"][qid]["text"],words[qid],dense[qid],method,profile)
                times.append(dict(query=qid,repeat=repeat,method=method,retrieval_ms=result["retrieval_ms"],rerank_ms=result["rerank_ms"]))
    db.close()
    report=dict(version="li-score/v1",records=records,timings=times,first_query_diagnostic=diagnostic,
                oracle_checks=checks,baseline_selection=selection,judgments=inputs["qrels"],
                protocol_sha256=digest(HERE/"recorded/protocol.json"),analysis_plan_sha256=digest(HERE/"recorded/analysis-plan.json"),
                scripts={n:digest(HERE/n) for n in ["benchmark.py","host_scorer.py","metrics.py","oracle.py"]},
                python=platform.python_version(),numpy=np.__version__,platform=platform.platform())
    with gzip.open(output,"wt") as stream:json.dump(report,stream,separators=(",",":"))
    print(output,flush=True)


def live_worker(output):
    from encoder import BGEEncoder,ColBERTEncoder
    inputs,_=load();protocol=inputs["protocol"];profile=inputs["manifest"]["profile_id"]
    db=connect();words=expressions(list(inputs["queries"].values()))
    start=time.perf_counter();bge=BGEEncoder(OLD);colbert=ColBERTEncoder()
    model_load_seconds=time.perf_counter()-start
    # Peak memory probe uses the same maximum-length batch shape as document preparation.
    documents=[row[0] for row in db.execute("SELECT body FROM sources ORDER BY length(body) DESC,id LIMIT 8")]
    probe=colbert.encode(documents,"document");probe_rows=[len(a) for a in probe];del probe
    times=[]
    for qid in protocol["timing_ids"][:3]:
        bge.encode([inputs["queries"][qid]["text"]]);colbert.encode([inputs["queries"][qid]["text"]],"query")
    for repeat in range(3):
        ids=protocol["timing_ids"][:];random.Random(991+repeat).shuffle(ids)
        for qid in ids:
            text=inputs["queries"][qid]["text"]
            methods=METHODS[:];random.Random(int(qid)+repeat).shuffle(methods)
            for method in methods:
                variants=[False,True] if repeat%2==0 else [True,False]
                for late in variants:
                    start=time.perf_counter_ns()
                    vector=bge.encode([text])[0] if method!="bm25" else np.zeros(384,dtype=np.float32)
                    bge_ms=(time.perf_counter_ns()-start)/1e6
                    t=time.perf_counter_ns();query=colbert.encode([text],"query")[0].tobytes() if late else None
                    colbert_ms=(time.perf_counter_ns()-t)/1e6
                    result=operation(db,qid,text,words[qid],vector,method,profile,late,query)
                    total=(time.perf_counter_ns()-start)/1e6
                    times.append(dict(query=qid,repeat=repeat,method=method,late=late,total_ms=total,
                                      bge_encode_ms=bge_ms,colbert_encode_ms=colbert_ms,
                                      retrieval_ms=result["retrieval_ms"],rerank_ms=result["rerank_ms"],ids=result["ids"]))
        print("live repetition",repeat+1,flush=True)
    db.close()
    report=dict(version="li-live/v1",model_load_seconds=model_load_seconds,document_batch_probe_rows=probe_rows,
                timings=times,protocol_sha256=digest(HERE/"recorded/protocol.json"),
                scripts={n:digest(HERE/n) for n in ["benchmark.py","host_scorer.py","encoder.py"]})
    with gzip.open(output,"wt") as stream:json.dump(report,stream,separators=(",",":"))
    print(output,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["score","live"])
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("output already exists")
    (score_worker if args.phase=="score" else live_worker)(args.output)
