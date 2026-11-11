"""Prepare, run and verify an isolated SciFact BlitzRank comparison."""
import argparse
import asyncio
import gzip
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sqlite3
import statistics
import subprocess
import sys
import time

from oracle import Budget, compare, permutation, usage

HERE = Path(__file__).resolve().parent
PRIOR = HERE / ".work/inputs/benchmarks-ann/experiments/2026-11-08-late-interaction"
UPSTREAM_REVISION = "076018cf6153608203029e29562c36f50273b348"
METHODS = ["bm25", "dense", "rrf"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(Path(path).read_text())


def write(path, value):
    if str(path).endswith(".gz"):
        with gzip.open(path, "xt") as stream:
            json.dump(value, stream, separators=(",", ":"))
    else:
        with Path(path).open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")


def prior_modules():
    sys.path.insert(0, str(PRIOR))
    import benchmark
    import encoder
    import metrics
    return benchmark, encoder, metrics


def connect(path, extension):
    db = sqlite3.connect(Path(path).as_uri() + "?mode=ro&immutable=1", uri=True, isolation_level=None)
    db.enable_load_extension(True)
    try:
        db.load_extension(str(extension))
    finally:
        db.enable_load_extension(False)
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA cache_size=-2048")
    db.execute("PRAGMA mmap_size=0")
    return db


def source_hashes():
    paths = [HERE / name for name in ["experiment.py", "oracle.py"]]
    paths += [PRIOR / name for name in ["benchmark.py", "encoder.py", "freeze.py", "build_baselines.py", "host_scorer.py", "metrics.py"]]
    return {str(p.relative_to(HERE.parent)): sha(p) for p in paths}


def prepare(args):
    benchmark, encoder, metric = prior_modules()
    upstream = HERE / ".work/upstream"
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION or subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True):
        raise ValueError("requires a clean checkout of the pinned upstream revision")
    previous = read(PRIOR / "recorded/protocol.json")
    from freeze import dataset, OLD
    docs, queries, judgments, _ = dataset()
    all_ids = previous["tuning_ids"] + previous["evaluation_ids"]
    vectors_path = PRIOR / ".work/dense-queries.npy"
    import numpy as np
    vectors = np.load(vectors_path, allow_pickle=False)
    if vectors.shape != (200, 384) or not np.isfinite(vectors).all():
        raise ValueError("invalid frozen dense query vectors")
    dense = dict(zip(all_ids, vectors))
    db_path = PRIOR / ".work/baseline-only.db"
    if any(Path(str(db_path) + suffix).exists() for suffix in ["-wal", "-journal"]):
        raise ValueError("immutable baseline must not have a WAL or rollback journal")
    if sha(benchmark.EXTENSION) != benchmark.EXTENSION_SHA:
        raise ValueError("native extension identity changed")
    db = connect(db_path, benchmark.EXTENSION)
    try:
        actual = dict(db.execute("SELECT id,body FROM sources"))
        expected = {int(d["_id"]): d["title"] + " " + d["text"] for d in docs}
        if actual != expected or db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("baseline source content or database integrity differs")
        words = benchmark.expressions([queries[q] for q in all_ids])
        prior_rows = {(r["query"], r["method"]): r for r in read(PRIOR / "recorded/score.json.gz")["records"]}
        rows = []
        for qid in all_ids:
            for method in METHODS:
                candidates = benchmark.first_stage(db, method, words[qid], dense[qid])
                if candidates != prior_rows[qid, method]["candidates"]:
                    raise ValueError("fresh baseline differs from retained SciFact candidates")
                candidates = candidates[:50]
                rows.append(dict(query=qid, method=method, candidates=candidates,
                                 ids=candidates[:10], metrics=metric.metrics(candidates[:10], judgments[qid], candidates)))
        evaluation = previous["timing_ids"]
        if args.evaluation_count == 150:
            evaluation = previous["evaluation_ids"]
        selected = max(METHODS, key=lambda m: statistics.mean(r["metrics"]["ndcg10"] for r in rows
                       if r["query"] in previous["tuning_ids"] and r["method"] == m))
        # Measure live baseline calls now; model load and warmup stay outside timers.
        bge = encoder.BGEEncoder(OLD)
        bge.encode([queries[previous["tuning_ids"][0]]["text"]])
        timing = []
        for repeat in range(3):
            order = [(q, m) for q in evaluation for m in METHODS]
            random.Random(20261111 + repeat).shuffle(order)
            for qid, method in order:
                start = time.perf_counter_ns()
                vector = bge.encode([queries[qid]["text"]])[0] if method != "bm25" else dense[qid]
                encoded = time.perf_counter_ns()
                candidates = benchmark.first_stage(db, method, words[qid], vector)[:50]
                elapsed = (time.perf_counter_ns() - start) / 1e6
                row = next(r for r in rows if r["query"] == qid and r["method"] == method)
                if candidates != row["candidates"]:
                    raise ValueError("live dense query changed candidate identities")
                timing.append(dict(query=qid, method=method, repeat=repeat, total_ms=elapsed,
                                   query_encode_ms=(encoded-start)/1e6))
        graph_dir = upstream / "blitzrank/engine/algorithms/tournament_graph"
        graph_files = ["__init__.py", "compare_oracle.py", "tournament_graph.py", "tournament_graph_sort.py", "types.py"]
        assets = [db_path, vectors_path, benchmark.EXTENSION, OLD/"scifact.zip", OLD/"model.onnx", OLD/"model-tokenizer.json"]
        protocol = dict(version="scifact-blitzrank/v1", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        model="gpt-6-luna", endpoint="https://api.openai.com/v1",
                        model_description="OpenAI hosted gpt-6-luna alias; no immutable dated snapshot listed",
                        upstream_revision=revision, upstream_files={n:sha(graph_dir/n) for n in graph_files},
                        scripts=source_hashes(), prior_protocol_sha256=sha(PRIOR/"recorded/protocol.json"),
                        assets={str(p):sha(p) for p in assets}, database=str(db_path), extension=str(benchmark.EXTENSION),
                        corpus_documents=len(docs), tuning_ids=previous["tuning_ids"], evaluation_ids=evaluation,
                        prior_evaluation_ids=previous["evaluation_ids"], selected_baseline=selected,
                        candidate_limit=50, rrf_branch_candidates=50, rrf_constant=60, topk=10, window_size=20,
                        max_document_tokens=1024, max_query_tokens=1024, tokenizer="o200k_base",
                        max_rounds=50, max_output_tokens=2048, call_timeout_seconds=120,
                        max_request_seconds=1800, max_run_tokens=20000000,
                        temperature="omitted", reasoning_effort="low", retries=0, api_cap_usd=5,
                        pricing=dict(input_usd_per_million=.1,cached_input_usd_per_million=.01,
                                     output_usd_per_million=.5,reserved_input_usd_per_million=.125,
                                     source="https://developers.openai.com/api/docs/models/gpt-6-luna",checked="2026-11-11"),
                        quality_gate=dict(ndcg10_gain=.02, bootstrap_low_above_zero=True, samples=10000, seed=20261111),
                        split_caveat="Previously examined SciFact training-label split; model contamination unknown. The 30-query pilot uses the previous timing subset, chosen before BlitzRank outcomes.",
                        latency_scope="Warm BGE model; full query encoding/retrieval/fetch/prompt/hosted inference/graph time. Three baseline repetitions, one rerank pass; no filesystem cache flush or concurrency benchmark.",
                        scope="Independent archive only; original baselines use 100 candidates (RRF 50 per branch), reranking admits the first 50. No adapter/core changes.")
        protocol["packages"] = {n:importlib.metadata.version(n) for n in ["numpy","networkx","onnxruntime","tokenizers","tiktoken"]}
        protocol["python"] = platform.python_version()
        protocol["platform"] = platform.platform()
        protocol["sqlite"] = db.execute("SELECT sqlite_version(),vec_version()").fetchone()
        target = HERE/"recorded"
        target.mkdir(exist_ok=False)
        write(target/"inputs.json.gz", dict(queries={q:queries[q]["text"] for q in all_ids},
                                           qrels={q:judgments[q] for q in all_ids}, documents=expected,
                                           expressions=words))
        write(target/"baselines.json.gz", dict(rows=rows, timings=timing))
        protocol["inputs_sha256"] = sha(target/"inputs.json.gz")
        protocol["baselines_sha256"] = sha(target/"baselines.json.gz")
        write(target/"protocol.json", protocol)
        print(json.dumps(dict(prepared=True, baseline_rows=len(rows), baseline_timings=len(timing),
                              evaluation_queries=len(evaluation), selected_baseline=selected)), flush=True)
    finally:
        db.close()


def verify_inputs(recorded, live=False):
    protocol = read(recorded/"protocol.json")
    if sha(recorded/"inputs.json.gz") != protocol["inputs_sha256"] or sha(recorded/"baselines.json.gz") != protocol["baselines_sha256"]:
        raise ValueError("retained input or baseline hash mismatch")
    for name, expected in protocol["scripts"].items():
        if sha(HERE.parent/name) != expected:
            raise ValueError("experiment source changed: " + name)
    if live:
        for name, expected in protocol["assets"].items():
            if sha(name) != expected:
                raise ValueError("experiment asset changed: " + name)
        base = HERE/".work/upstream/blitzrank/engine/algorithms/tournament_graph"
        for name, expected in protocol["upstream_files"].items():
            if sha(base/name) != expected:
                raise ValueError("upstream algorithm changed: " + name)
    return protocol, read(recorded/"inputs.json.gz"), read(recorded/"baselines.json.gz")


def run(args):
    recorded = HERE/"recorded"
    protocol, inputs, baseline = verify_inputs(recorded, live=True)
    benchmark, encoder, _ = prior_modules()
    import tiktoken
    tokenizer = tiktoken.get_encoding(protocol["tokenizer"])
    sys.path.insert(0, str(HERE/".work/upstream/blitzrank/engine/algorithms"))
    from tournament_graph.compare_oracle import CompareOracle, Item, OracleResult, make_edges_from_linear_order
    from tournament_graph.tournament_graph_sort import tournament_graph_sort_async, TournamentGraphSortConfig
    from freeze import OLD
    bge = encoder.BGEEncoder(OLD)
    bge.encode([inputs["queries"][protocol["tuning_ids"][0]]])
    db = connect(protocol["database"], protocol["extension"])
    results = recorded/"results"
    results.mkdir(exist_ok=True)
    budget = Budget(recorded/"budget.json")
    order = [(q, m) for q in protocol["evaluation_ids"] for m in METHODS]
    random.Random(20261111).shuffle(order)
    spent = 0
    for path in results.glob("*.json.gz"):
        for call in read(path)["calls"]:
            spent += usage(call["response"])["total_tokens"]
    completed_now = 0
    try:
        for qid, method in order:
            target = results/f"{qid}-{method}.json.gz"
            if target.exists():
                continue
            if args.max_tasks is not None and completed_now >= args.max_tasks:
                break
            if spent >= protocol["max_run_tokens"]:
                raise RuntimeError("frozen run token budget exhausted")
            expected = next(r for r in baseline["rows"] if r["query"] == qid and r["method"] == method)
            calls, rounds, trimmed = [], [], []
            record = dict(query=qid, method=method, protocol_sha256=sha(recorded/"protocol.json"),
                          calls=calls, rounds=rounds, status="failed")
            # The append log survives interruption before the complete result is written.
            journal = results/f"{qid}-{method}.calls.partial"
            if journal.exists():
                raise RuntimeError(f"interrupted request retained at {journal}; inspect it before a new run")
            with journal.open("x") as log:
                def save_call(call):
                    calls.append(call)
                    log.write(json.dumps(call) + "\n")
                    log.flush()
                    os.fsync(log.fileno())

                start = time.perf_counter_ns()
                try:
                    vector = bge.encode([inputs["queries"][qid]])[0] if method != "bm25" else None
                    encoded = time.perf_counter_ns()
                    candidates = benchmark.first_stage(db, method, inputs["expressions"][qid], vector)[:protocol["candidate_limit"]]
                    if candidates != expected["candidates"]:
                        raise ValueError("live candidates differ from frozen baseline")
                    documents = dict(db.execute("SELECT id,body FROM sources WHERE id IN (" + ",".join("?" for _ in candidates) + ")", candidates))
                    fetched = time.perf_counter_ns()
                    prepared = {}
                    for key in candidates:
                        tokens = tokenizer.encode(documents[key].replace("Title: Content: ", "").strip(), disallowed_special=())
                        if len(tokens) > protocol["max_document_tokens"]:
                            trimmed.append(key)
                        prepared[key] = tokenizer.decode(tokens[:protocol["max_document_tokens"]])
                    query = tokenizer.decode(tokenizer.encode(inputs["queries"][qid].strip(), disallowed_special=())[:protocol["max_query_tokens"]])

                    class ModelOracle(CompareOracle):
                        async def _compare_k_async(self, items):
                            if (time.perf_counter_ns()-start)/1e9 > protocol["max_request_seconds"]:
                                raise RuntimeError("request deadline exhausted between comparisons")
                            if spent + sum(usage(c["response"])["total_tokens"] for c in calls) >= protocol["max_run_tokens"]:
                                raise RuntimeError("run token budget exhausted between comparisons")
                            indices, tokens = compare(protocol, query, [prepared[i.id] for i in items], save_call, budget=budget)
                            return OracleResult(make_edges_from_linear_order([items[i] for i in indices]), tokens)

                    oracle = ModelOracle(min(protocol["window_size"], len(candidates)))
                    result = asyncio.run(tournament_graph_sort_async(
                        [Item(key) for key in candidates], oracle, min(protocol["topk"], len(candidates)),
                        TournamentGraphSortConfig(max_num_rounds=protocol["max_rounds"], on_round_complete=rounds.append)))
                    ids = [item.id for item in result.results]
                    if len(ids) != min(10, len(candidates)) or len(set(ids)) != len(ids) or not set(ids) <= set(candidates):
                        raise ValueError("invalid final candidate identities")
                    record.update(status="ok", ids=ids, candidates=candidates,
                                  query_encode_ms=(encoded-start)/1e6, retrieval_fetch_ms=(fetched-encoded)/1e6,
                                  scc_stats=vars(result.get_scc_stats()), trimmed_document_ids=trimmed)
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                record["total_ms"] = (time.perf_counter_ns()-start)/1e6
                write(target, record)
            journal.unlink()
            spent += sum(usage(c["response"])["total_tokens"] for c in calls)
            completed_now += 1
            print(json.dumps(dict(query=qid, method=method, status=record["status"],
                                  calls=len(calls), seconds=record["total_ms"]/1000, run_tokens=spent,
                                  error=record.get("error"))), flush=True)
    finally:
        db.close()


def summary(recorded):
    protocol, inputs, baseline = verify_inputs(recorded)
    _, _, metric = prior_modules()
    expected = {(q, m) for q in protocol["evaluation_ids"] for m in METHODS}
    records = [read(p) for p in sorted((recorded/"results").glob("*.json.gz"))]
    keys = [(r["query"], r["method"]) for r in records]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError("duplicate or unexpected result identities")
    lookup = {(r["query"], r["method"]): r for r in baseline["rows"]}
    for row in baseline["rows"]:
        if row["ids"] != row["candidates"][:10] or len(row["candidates"]) != len(set(row["candidates"])):
            raise ValueError("invalid retained baseline identities")
        if metric.metrics(row["ids"], inputs["qrels"][row["query"]], row["candidates"]) != row["metrics"]:
            raise ValueError("baseline relevance metrics differ")
    totals = dict(input_tokens=0, output_tokens=0, reasoning_tokens=0, total_tokens=0)
    for row in records:
        if row["protocol_sha256"] != sha(recorded/"protocol.json"):
            raise ValueError("result protocol mismatch")
        if not math.isfinite(row["total_ms"]) or row["total_ms"] < 0:
            raise ValueError("invalid request duration")
        for call in row["calls"]:
            for key, value in usage(call["response"]).items():
                totals[key] += value
            if row["status"] == "ok":
                permutation(call["response"]["choices"][0]["message"]["content"], (len(call["request"]["messages"])-4)//2)
        if row["status"] == "ok":
            if not row["calls"] or len(row["calls"]) != len(row["rounds"]):
                raise ValueError("successful reranking requires one logged model call per round")
            if row["candidates"] != lookup[row["query"],row["method"]]["candidates"] or not set(row["ids"]) <= set(row["candidates"]):
                raise ValueError("reranked candidates changed")
            replay(row, protocol)
    complete = set(keys) == expected and all(r["status"] == "ok" for r in records)
    result = dict(complete=complete, expected_requests=len(expected), recorded_requests=len(records),
                  successful_requests=sum(r["status"] == "ok" for r in records), tokens=totals,
                  methods={}, failures=[{k:r[k] for k in ["query","method","error"]} for r in records if r["status"] != "ok"],
                  caveat=protocol["split_caveat"])
    for method in METHODS:
        base = [lookup[q,method] for q in protocol["evaluation_ids"]]
        times = [r["total_ms"] for r in baseline["timings"] if r["method"] == method]
        entry = dict(baseline={k:statistics.mean(r["metrics"][k] for r in base) for k in base[0]["metrics"]},
                     baseline_total_ms={"p50":metric.percentile(times,.5),"p95":metric.percentile(times,.95)})
        rows = [r for r in records if r["method"] == method and r["status"] == "ok"]
        if len(rows) == len(base):
            scored = [metric.metrics(r["ids"], inputs["qrels"][r["query"]], r["candidates"]) for r in rows]
            entry["blitzrank"] = {k:statistics.mean(r[k] for r in scored) for k in scored[0]}
            entry["paired_ndcg10"] = metric.bootstrap([m["ndcg10"]-lookup[r["query"],method]["metrics"]["ndcg10"] for m,r in zip(scored,rows)], seed=20261111)
            entry["blitzrank_total_ms"] = {"p50":metric.percentile([r["total_ms"] for r in rows],.5), "p95":metric.percentile([r["total_ms"] for r in rows],.95)}
            entry["mean_tokens"] = statistics.mean(sum(usage(c["response"])["total_tokens"] for c in r["calls"]) for r in rows)
            entry["mean_calls"] = statistics.mean(len(r["calls"]) for r in rows)
        result["methods"][method] = entry
    if complete:
        rrf = {r["query"]:r for r in records if r["method"] == "rrf"}
        differences = [metric.metrics(rrf[q]["ids"],inputs["qrels"][q])["ndcg10"]-lookup[q,protocol["selected_baseline"]]["metrics"]["ndcg10"] for q in protocol["evaluation_ids"]]
        result["primary_vs_selected_baseline"] = metric.bootstrap(differences, seed=20261111)
        result["quality_gate_pass"] = statistics.mean(differences) >= .02 and result["primary_vs_selected_baseline"]["low"] > 0
    if (recorded/"budget.json").exists():
        ledger = read(recorded/"budget.json")
        charged = sum(e["charged_nusd"] for e in ledger["entries"])
        if charged > ledger["cap_nusd"]:
            raise ValueError("budget cap exceeded")
        result["budget"] = dict(cap_usd=ledger["cap_nusd"]/1e9,
                                conservative_charge_usd=charged/1e9,
                                requests=len(ledger["entries"]),unsettled_requests=sum(not e["settled"] for e in ledger["entries"]))
        cached = sum(c["response"]["usage"].get("prompt_tokens_details",{}).get("cached_tokens",0) for r in records for c in r["calls"])
        result["standard_price_estimate_usd"] = ((totals["input_tokens"]-cached)*100+cached*10+totals["output_tokens"]*500)/1e9
    return result


def replay(row, protocol):
    """Rebuild the actual upstream graph from retained model decisions."""
    base = HERE/".work/upstream/blitzrank/engine/algorithms/tournament_graph"
    for name, expected in protocol["upstream_files"].items():
        if sha(base/name) != expected:
            raise ValueError("upstream replay source changed: " + name)
    sys.path.insert(0,str(base.parent))
    from tournament_graph.compare_oracle import CompareOracle, Item, OracleResult, make_edges_from_linear_order
    from tournament_graph.tournament_graph_sort import tournament_graph_sort_async, TournamentGraphSortConfig

    class RecordedOracle(CompareOracle):
        async def _compare_k_async(self, items):
            index = self.calls - 1
            call = row["calls"][index]
            if row["rounds"][index]["nodes_for_match"] != [str(i.id) for i in items]:
                raise ValueError("logged comparisons do not match the upstream schedule")
            if call["request"]["model"] != protocol["model"] or call["response"]["model"] != protocol["model"]:
                raise ValueError("comparison model identity differs")
            choice = call["response"]["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("successful receipt contains an unfinished model generation")
            indices = permutation(choice["message"]["content"],len(items))
            return OracleResult(make_edges_from_linear_order([items[i] for i in indices]))

    oracle = RecordedOracle(min(protocol["window_size"],len(row["candidates"])))
    result = asyncio.run(tournament_graph_sort_async([Item(i) for i in row["candidates"]],oracle,
                        min(protocol["topk"],len(row["candidates"])),
                        TournamentGraphSortConfig(max_num_rounds=protocol["max_rounds"])))
    if [i.id for i in result.results] != row["ids"] or oracle.calls != len(row["calls"]):
        raise ValueError("replayed graph ranking or call count differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--evaluation-count", choices=[30,150], type=int, default=30)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--max-tasks", type=int)
    commands.add_parser("analyze")
    args = parser.parse_args()
    if not __debug__:
        parser.error("run without -O; upstream uses assertions")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        if args.max_tasks is not None and args.max_tasks < 1:
            parser.error("--max-tasks must be positive")
        with (HERE/".work/run.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)
    else:
        result = summary(HERE/"recorded")
        print(json.dumps(result, indent=2))
        if not result["complete"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
