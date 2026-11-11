"""Freeze SciFact splits and acceptance budgets before any ranking outcomes."""
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "2026-10-26-scifact-hybrid/.work"
DATA_SHA = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dataset():
    assert digest(OLD / "scifact.zip") == DATA_SHA
    with zipfile.ZipFile(OLD / "scifact.zip") as archive:
        docs = sorted((json.loads(s) for s in archive.read("scifact/corpus.jsonl").splitlines()), key=lambda d: int(d["_id"]))
        queries = {r["_id"]: r for r in (json.loads(s) for s in archive.read("scifact/queries.jsonl").splitlines())}
        qrels = {}
        for row in csv.DictReader(io.StringIO(archive.read("scifact/qrels/train.tsv").decode()), delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
        seen = {r["query-id"] for r in csv.DictReader(io.StringIO(archive.read("scifact/qrels/test.tsv").decode()), delimiter="\t")}
    assert len(docs) == 5183 and len(qrels) == 809 and len(seen) == 300
    assert not (set(qrels) & seen)
    assert all(grade == 1 for rows in qrels.values() for grade in rows.values())
    return docs, queries, qrels, seen


def main():
    docs, queries, qrels, seen = dataset()
    ids = sorted(qrels, key=int)
    random.Random(20261108).shuffle(ids)
    tuning, evaluation = ids[:50], ids[50:200]
    timed = random.Random(20261109).sample(evaluation, 30)
    model_manifest = json.loads((HERE / ".work/model/manifest.json").read_text())
    old_manifest = json.loads((OLD / "data-manifest.json").read_text())
    for name in ["documents.npy", "model.onnx", "model-tokenizer.json"]:
        assert digest(OLD / name) == old_manifest["files"][name]
    protocol = dict(
        version="late-interaction-scifact/v1", frozen_before_ranking=True,
        split_description="Local deterministic split of BEIR SciFact TRAIN annotations; not original blind test. Prior 300 test-query IDs excluded.",
        dataset_sha256=DATA_SHA, corpus_documents=len(docs), train_labelled_queries=len(qrels),
        tuning_ids=tuning, evaluation_ids=evaluation, timing_ids=timed,
        excluded_prior_test_ids=sorted(seen, key=int), split_seed=20261108, timing_seed=20261109,
        profile=dict(model=model_manifest["model"], revision=model_manifest["revision"],
                     weights_sha256=model_manifest["files"]["model.safetensors"]["sha256"],
                     encoder_source_revision=model_manifest["source_revision"],
                     dimensions=128, query_maxlen=32, doc_maxlen=180,
                     attend_to_mask_tokens=False, mask_document_punctuation=True,
                     query_marker="[unused0]", document_marker="[unused1]",
                     input="title + single space + text", normalization="L2 float32 rows",
                     score="maxsim_normalized_dot_v1", scalar_accumulator="float64", device="cpu"),
        dense=dict(model=old_manifest["model"], revision=old_manifest["revision"],
                   document_vectors_sha256=old_manifest["files"]["documents.npy"],
                   document_runtime=old_manifest["onnxruntime"], query_instruction=old_manifest["query_instruction"]),
        candidates=dict(bm25=100, dense=100, rrf_branch=50, rrf_constant=60, rerank_all_admitted=True),
        threads=dict(encoder=4, onnx_intra=4, onnx_inter=1, blas=1),
        request_limits=dict(candidates=100, query_tokens=128, document_tokens=4096, dimensions=4096,
                            query_bytes=1048576, document_bytes=8388608, aggregate_document_bytes=33554432,
                            work=134217728),
        acceptance=dict(ndcg10_absolute_gain=0.02, paired_bootstrap_lower_bound_above_zero=True,
                        baseline_selection="highest tuning nDCG@10, tie bm25/dense/rrf order",
                        reranking_p95_ms=250, measured_end_to_end_p95_ms=500,
                        scoring_process_peak_rss_bytes=268435456, model_process_peak_rss_bytes=2147483648,
                        final_sqlite_store_bytes=1073741824, bootstrap_seed=20261110, bootstrap_samples=10000),
        timing=dict(repetitions=3, queries=30, first_query_reported_separately=True,
                    warmup_queries=3, filesystem_cache_flushed=False,
                    process_peak_rss="fresh child RUSAGE_CHILDREN on macOS; includes imports and SQLite/model state"),
        inference_provenance="Pinned public weights, no remote model code, no custom MaxSim C/C++ extension; encoder implementation checked against official source.",
        limitations=["Public training annotations used as local evaluation; model/data contamination not independently audited.",
                     "Single text checkpoint and document truncation at checkpoint's 180-token limit; no multimodal claim.",
                     "First-stage exact dense/BM25/RRF over the same full corpus; reranking cannot recover absent candidates.",
                     "Budgets are preregistered acceptance targets, not observed performance promises."])
    target = HERE / "recorded/protocol.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(protocol, indent=2) + "\n"
    if target.exists():
        assert target.read_text() == text, "frozen protocol differs; use a new experiment, never overwrite"
    else:
        with target.open("x") as output:
            output.write(text)
    print(target, digest(target))


if __name__ == "__main__":
    main()
