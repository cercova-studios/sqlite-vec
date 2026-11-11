"""Explicitly encode frozen SciFact inputs on CPU into a fresh experiment database."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import time

import numpy as np

from encoder import BGEEncoder, ColBERTEncoder
from freeze import HERE, OLD, dataset, digest


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def artifact_hash(profile, role, input_sha, matrix):
    shape = np.asarray([matrix.shape[0], matrix.shape[1]], dtype="<u4").tobytes()
    return hashlib.sha256(b"sqlite-vec/li-matrix/v1\0" + bytes.fromhex(profile) + bytes([role == "query"])
                          + bytes.fromhex(input_sha) + shape + matrix.tobytes()).hexdigest()


def main():
    start = time.perf_counter()
    work = HERE / ".work"
    protocol_path = HERE / "recorded/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    docs, queries, qrels, seen = dataset()
    qids = protocol["tuning_ids"] + protocol["evaluation_ids"]
    db_path = work / "corpus.db"
    if db_path.exists():
        raise FileExistsError("fresh database required; retain prior attempts explicitly")
    profile_json = canonical(protocol["profile"])
    profile = hashlib.sha256(profile_json).hexdigest()
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""CREATE TABLE li_profiles(id TEXT PRIMARY KEY,metadata TEXT NOT NULL);
        CREATE TABLE sources(id INTEGER PRIMARY KEY,body TEXT NOT NULL,source_revision INTEGER NOT NULL);
        CREATE TABLE li_artifacts(artifact_id INTEGER PRIMARY KEY,sha256 TEXT UNIQUE NOT NULL,
          profile_id TEXT NOT NULL,role TEXT NOT NULL,token_count INTEGER NOT NULL,dimensions INTEGER NOT NULL,
          input_sha256 TEXT NOT NULL,vectors BLOB NOT NULL);
        CREATE TABLE li_refs(role TEXT NOT NULL,item_id INTEGER NOT NULL,artifact_sha256 TEXT NOT NULL,
          source_revision INTEGER NOT NULL,PRIMARY KEY(role,item_id)) WITHOUT ROWID;""")
    db.execute("INSERT INTO li_profiles VALUES(?,?)", (profile, profile_json.decode()))
    db.executemany("INSERT INTO sources VALUES(?,?,1)", [(int(d["_id"]), d["title"] + " " + d["text"]) for d in docs])
    db.commit()
    bge = BGEEncoder(OLD)
    qvectors = []
    bge_times = {}
    for qid in qids:
        t = time.perf_counter_ns()
        qvectors.append(bge.encode([queries[qid]["text"]])[0])
        bge_times[qid] = (time.perf_counter_ns() - t) / 1e6
    np.save(work / "dense-queries.npy", np.asarray(qvectors, dtype=np.float32))
    del bge
    encoder = ColBERTEncoder()
    qualification = encoder.qualification()
    stats = dict(document_truncated=0, query_truncated=0, matrix_bytes=0, artifact_insert_ms=0,
                 document_encode_ms=0, query_encode_ms={}, document_tokens=[])

    def store(role, key, text, matrix):
        input_sha = hashlib.sha256(text.encode()).hexdigest()
        hashed = artifact_hash(profile, role, input_sha, matrix)
        data = matrix.tobytes()
        db.execute("INSERT OR IGNORE INTO li_artifacts(sha256,profile_id,role,token_count,dimensions,input_sha256,vectors) VALUES(?,?,?,?,?,?,?)",
                   (hashed, profile, role, len(matrix), matrix.shape[1], input_sha, data))
        prior = db.execute("SELECT vectors FROM li_artifacts WHERE sha256=?", (hashed,)).fetchone()[0]
        assert prior == data
        db.execute("INSERT INTO li_refs VALUES(?,?,?,1)", (role, int(key), hashed))
        stats["matrix_bytes"] += len(data)

    # Model inference happens before each short artifact-insertion transaction.
    for offset in range(0, len(docs), 8):
        batch = docs[offset:offset + 8]
        texts = [d["title"] + " " + d["text"] for d in batch]
        stats["document_truncated"] += sum(len(encoder.tokenizer.encode(t)) + 1 > 180 for t in texts)
        t = time.perf_counter_ns()
        matrices = encoder.encode(texts, "document")
        stats["document_encode_ms"] += (time.perf_counter_ns() - t) / 1e6
        t = time.perf_counter_ns()
        with db:
            for doc, text, matrix in zip(batch, texts, matrices):
                store("document", doc["_id"], text, matrix)
                stats["document_tokens"].append(len(matrix))
        stats["artifact_insert_ms"] += (time.perf_counter_ns() - t) / 1e6
        if offset % 256 == 0:
            print(f"Documents {min(offset+8,len(docs))}/{len(docs)}; elapsed {time.perf_counter()-start:.1f}s", flush=True)
    for qid in qids:
        text = queries[qid]["text"]
        stats["query_truncated"] += len(encoder.tokenizer.encode(text)) + 1 > 32
        t = time.perf_counter_ns()
        matrix = encoder.encode([text], "query")[0]
        stats["query_encode_ms"][qid] = (time.perf_counter_ns() - t) / 1e6
        with db:
            store("query", qid, text, matrix)
    assert db.execute("SELECT count(*) FROM li_refs WHERE role='document'").fetchone()[0] == 5183
    assert db.execute("SELECT count(*) FROM li_refs WHERE role='query'").fetchone()[0] == 200
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    selected = dict(queries=[queries[qid] for qid in qids], qrels={qid: qrels[qid] for qid in qids})
    (work / "selected.json").write_text(json.dumps(selected) + "\n")
    import torch, transformers, onnxruntime
    manifest = dict(protocol_sha256=digest(protocol_path), profile_id=profile, qualification=qualification,
                    stats=stats, bge_query_encode_ms=bge_times, seconds=time.perf_counter()-start,
                    stage_db_bytes=db_path.stat().st_size, stage_db_sha256=digest(db_path),
                    platform=platform.platform(), python=platform.python_version(),
                    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,onnxruntime=onnxruntime.__version__),
                    files={"dense-queries.npy":digest(work/"dense-queries.npy"),"selected.json":digest(work/"selected.json")},
                    script_hashes={n:digest(HERE/n) for n in ["prepare.py","encoder.py","freeze.py","acquire.py"]})
    (work / "embedding-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(work / "embedding-manifest.json", flush=True)


if __name__ == "__main__":
    main()
