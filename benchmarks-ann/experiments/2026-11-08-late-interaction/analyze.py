"""Verify retained receipts without models, then summarize the frozen experiment."""
import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

from metrics import bootstrap, percentile

HERE = Path(__file__).resolve().parent
METHODS = ["bm25", "dense", "rrf"]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def independent_metrics(ids, judgments, candidates):
    positive = {int(k) for k, v in judgments.items() if v > 0}
    assert positive and len(ids) == len(set(ids)) and len(ids) <= 10
    hits = [i + 1 for i, key in enumerate(ids) if key in positive]
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(10, len(positive)) + 1))
    return dict(ndcg10=sum(1 / math.log2(i + 1) for i in hits) / ideal,
                mrr10=1 / hits[0] if hits else 0.0, recall10=len(hits) / len(positive),
                candidate_recall=len(set(candidates) & positive) / len(positive))


def verify_rows(score, live, protocol, inventory):
    """Check identities, coverage, rankings, independent metrics and work receipts."""
    tuning, evaluation, timing = (protocol[k + "_ids"] for k in ["tuning", "evaluation", "timing"])
    assert len(set(tuning)) == 50 and len(set(evaluation)) == 150 and len(set(timing)) == 30
    assert not set(tuning) & set(evaluation)
    assert not set(tuning + evaluation) & set(protocol["excluded_prior_test_ids"])
    assert set(timing) <= set(evaluation)
    assert inventory["query_tokens"] == protocol["profile"]["query_maxlen"] == 32
    assert inventory["dimensions"] == protocol["profile"]["dimensions"] == 128
    expected = {(q, split, method) for split, ids in [("tuning", tuning), ("evaluation", evaluation)]
                for q in ids for method in METHODS}
    assert len(score["records"]) == len(expected) == 600
    assert {(r["query"], r["split"], r["method"]) for r in score["records"]} == expected
    counts = {int(key): value for key, value in inventory["document_tokens"].items()}
    assert len(counts) == protocol["corpus_documents"]
    lookup = {}
    for r in score["records"]:
        candidates, scoring = r["candidates"], r["scoring"]
        assert 1 <= len(candidates) <= 100 and len(candidates) == len(set(candidates))
        assert set(candidates) <= counts.keys()
        assert r["baseline_ids"] == candidates[:10]
        pairs = scoring["scores"]
        assert len(pairs) == len(candidates) and {key for key, _ in pairs} == set(candidates)
        assert all(math.isfinite(value) for _, value in pairs)
        assert pairs == sorted(pairs, key=lambda pair: (-pair[1], pair[0]))
        assert r["late_ids"] == [key for key, _ in pairs[:10]]
        for variant in ["baseline", "late"]:
            actual = independent_metrics(r[variant + "_ids"], score["judgments"][r["query"]], candidates)
            assert all(abs(actual[k] - r[variant][k]) < 1e-14 for k in actual)
        m, d = 32, 128
        payload = sum(counts[key] * d * 4 for key in candidates)
        products = sum(m * counts[key] * d for key in candidates)
        work = m*d + sum(m*d + counts[key]*d + m*counts[key]*d + m*counts[key] + m for key in candidates)
        assert scoring["work_used"] == work <= protocol["request_limits"]["work"]
        assert scoring["component_products"] == products
        assert scoring["candidate_bytes"] == payload <= protocol["request_limits"]["aggregate_document_bytes"]
        assert scoring["matrix_bytes_read"] == payload + m*d*4
        assert scoring["scored_candidates"] == len(candidates)
        lookup[r["query"], r["method"]] = r
    averages = {method: statistics.mean(r["baseline"]["ndcg10"] for r in score["records"]
                                      if r["split"] == "tuning" and r["method"] == method) for method in METHODS}
    selection = score["baseline_selection"]
    assert selection["baseline"] == max(METHODS, key=lambda method: averages[method])
    assert all(abs(averages[k] - selection["tuning_ndcg10"][k]) < 1e-14 for k in METHODS)
    assert selection["primary_late"] == "rrf_late"
    assert len(score["oracle_checks"]) == 6
    for row in score["oracle_checks"]:
        assert row["host"] == row["oracle"]
        assert row["query"] in tuning[:3]
        assert row["document"] in lookup[row["query"], "rrf"]["candidates"][:2]
        assert dict(lookup[row["query"], "rrf"]["scoring"]["scores"])[row["document"]] == row["oracle"]
        m, n, d = 32, counts[row["document"]], 128
        assert row["work"] == m*d+n*d+m*n*d+m*n+m
    for data, variants in [(score, None), (live, [False, True])]:
        expected = set(itertools.product(timing, range(3), METHODS))
        if variants is not None:
            expected = {(*key, variant) for key in expected for variant in variants}
        found = [(r["query"], r["repeat"], r["method"]) + ((r["late"],) if variants is not None else ())
                 for r in data["timings"]]
        assert len(found) == len(set(found)) == len(expected) and set(found) == expected
        for r in data["timings"]:
            assert all(math.isfinite(r[k]) and r[k] >= 0 for k in r if k.endswith("_ms"))
            if variants is not None:
                assert r["ids"] == lookup[r["query"], r["method"]]["late_ids" if r["late"] else "baseline_ids"]
                assert r["total_ms"] >= r["bge_encode_ms"] + r["colbert_encode_ms"] + r["retrieval_ms"] + r["rerank_ms"]
    return lookup


def main():
    if not __debug__:
        raise RuntimeError("verification requires Python assertions; omit -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, help="also hash the retained local corpus SQLite file")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    recorded = HERE / "recorded"
    manifest = json.loads((recorded / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert sha(HERE / name) == expected, name
    load = lambda name: json.loads((recorded / name).read_text())
    protocol = load("protocol.json")
    score = json.loads(gzip.decompress((recorded / "score.json.gz").read_bytes()))
    live = json.loads(gzip.decompress((recorded / "live.json.gz").read_bytes()))
    for data in [score, live]:
        assert data["protocol_sha256"] == sha(recorded / "protocol.json")
        for name, expected in data["scripts"].items():
            assert sha(HERE / name) == expected, name
    assert score["analysis_plan_sha256"] == sha(recorded / "analysis-plan.json")
    baseline = load("baseline-manifest.json")
    embedding = load("embedding-manifest.json")
    assert baseline["script_sha256"] == sha(HERE / "build_baselines.py")
    for name, expected in embedding["script_hashes"].items():
        assert sha(HERE / name) == expected, name
    assert baseline["protocol_sha256"] == embedding["protocol_sha256"] == sha(recorded / "protocol.json")
    profile_json = json.dumps(protocol["profile"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(profile_json).hexdigest() == embedding["profile_id"]
    if args.database:
        assert sha(args.database) == baseline["combined_db_sha256"]
    lookup = verify_rows(score, live, protocol, load("inventory.json"))
    selected = score["baseline_selection"]["baseline"]
    primary = bootstrap([lookup[q, "rrf"]["late"]["ndcg10"] - lookup[q, selected]["baseline"]["ndcg10"]
                         for q in protocol["evaluation_ids"]])
    quality, timing = {}, {}
    for method in METHODS:
        rows = [lookup[q, method] for q in protocol["evaluation_ids"]]
        quality[method] = {v: {k: statistics.mean(r[v][k] for r in rows) for k in rows[0][v]}
                           for v in ["baseline", "late"]}
        quality[method]["paired_same_candidates"] = bootstrap([r["late"]["ndcg10"]-r["baseline"]["ndcg10"] for r in rows])
        times = [r for r in score["timings"] if r["method"] == method]
        timing[method] = {k: {"p50": percentile([r[k] for r in times], .5),
                              "p95": percentile([r[k] for r in times], .95)} for k in ["retrieval_ms", "rerank_ms"]}
        for late in [False, True]:
            times = [r for r in live["timings"] if r["method"] == method and r["late"] == late]
            timing[method]["live_late" if late else "live_baseline"] = {
                k: {"p50": percentile([r[k] for r in times], .5), "p95": percentile([r[k] for r in times], .95)}
                for k in ["total_ms", "bge_encode_ms", "colbert_encode_ms", "retrieval_ms", "rerank_ms"]}
    resources = {phase: load(phase + "-resources.json") for phase in ["score", "live"]}
    assert all(r["exit_code"] == 0 and r["peak_rss_bytes"] > 0 for r in resources.values())
    for phase, resource in resources.items():
        assert resource["log_sha256"] == sha(recorded / (phase + ".log"))
    acceptance = protocol["acceptance"]
    gates = dict(quality_gain=primary["mean"] >= acceptance["ndcg10_absolute_gain"],
                 quality_confidence=primary["low"] > 0,
                 rerank_latency=timing["rrf"]["rerank_ms"]["p95"] <= acceptance["reranking_p95_ms"],
                 live_latency=timing["rrf"]["live_late"]["total_ms"]["p95"] <= acceptance["measured_end_to_end_p95_ms"],
                 scoring_memory=resources["score"]["peak_rss_bytes"] <= acceptance["scoring_process_peak_rss_bytes"],
                 model_memory=resources["live"]["peak_rss_bytes"] <= acceptance["model_process_peak_rss_bytes"],
                 storage=baseline["combined_db_bytes"] <= acceptance["final_sqlite_store_bytes"])
    summary = dict(selected_baseline=selected, primary=primary, quality=quality, latency_ms=timing,
                   resources=resources, gates=gates, all_gates_pass=all(gates.values()),
                   storage=dict(baseline_bytes=baseline["baseline_only_bytes"], combined_bytes=baseline["combined_db_bytes"],
                                amplification=baseline["combined_db_bytes"]/baseline["baseline_only_bytes"]),
                   verification="600 quality rows, 810 timing rows, independent metrics/work/coverage, six real scalar pairs, input/script hashes")
    if args.write_summary:
        (recorded / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    else:
        assert summary == load("summary.json"), "retained summary differs from independently recomputed values"
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
