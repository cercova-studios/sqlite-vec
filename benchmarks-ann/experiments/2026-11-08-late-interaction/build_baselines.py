"""Add frozen SQLite BM25/exact-dense baselines to the immutable token fixture."""
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from freeze import HERE, OLD, dataset, digest

EXTENSION = HERE.parents[2] / "dist/bundles/949073f-default/lib/vec0.dylib"
EXTENSION_SHA = "4785d6f2ef2b1da23ed8c99f218b9dd519f4270e0789a3a20d52f8727f4c0261"


def connect(path):
    assert digest(EXTENSION) == EXTENSION_SHA
    db = sqlite3.connect(path, isolation_level=None)
    db.enable_load_extension(True)
    try:
        db.load_extension(str(EXTENSION))
    finally:
        db.enable_load_extension(False)
    return db


def populate(db, docs, vectors, sources_exist):
    start = time.perf_counter()
    db.execute("BEGIN IMMEDIATE")
    try:
        if not sources_exist:
            db.execute("CREATE TABLE sources(id INTEGER PRIMARY KEY,body TEXT NOT NULL,source_revision INTEGER NOT NULL)")
            db.executemany("INSERT INTO sources VALUES(?,?,1)", [(int(d["_id"]),d["title"]+' '+d["text"]) for d in docs])
        db.execute("CREATE VIRTUAL TABLE vec_docs USING vec0(embedding float[384])")
        db.execute("CREATE VIRTUAL TABLE fts_docs USING fts5(body)")
        for doc, vector in zip(docs, vectors):
            key = int(doc["_id"])
            db.execute("INSERT INTO vec_docs(rowid,embedding) VALUES(?,?)", (key, vector.tobytes()))
            db.execute("INSERT INTO fts_docs(rowid,body) VALUES(?,?)", (key, doc["title"]+' '+doc["text"]))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    return time.perf_counter()-start


def main():
    work = HERE / ".work"
    docs, _, _, _ = dataset()
    embedding = json.loads((work / "embedding-manifest.json").read_text())
    assert digest(work / "corpus.db") == embedding["stage_db_sha256"]
    protocol = json.loads((HERE / "recorded/protocol.json").read_text())
    assert digest(OLD / "documents.npy") == protocol["dense"]["document_vectors_sha256"]
    vectors = np.load(OLD / "documents.npy")
    assert vectors.shape == (5183,384) and np.isfinite(vectors).all()
    only = work / "baseline-only.db"
    if only.exists():
        raise FileExistsError("fresh baseline database required")
    db = connect(only)
    baseline_seconds = populate(db, docs, vectors, False)
    db.close()
    db = connect(work / "corpus.db")
    seconds = populate(db, docs, vectors, True)
    pages = dict(db.execute("SELECT name,sum(pgsize) FROM dbstat GROUP BY name"))
    options = [r[0] for r in db.execute("PRAGMA compile_options")]
    version = db.execute("SELECT sqlite_version(),vec_version()").fetchone()
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    report = dict(extension=str(EXTENSION), extension_sha256=EXTENSION_SHA,
                  sqlite=version[0], vec_version=version[1], compile_options=options,
                  baseline_build_seconds=baseline_seconds, combined_baseline_append_seconds=seconds,
                  baseline_only_bytes=only.stat().st_size, combined_db_bytes=(work/"corpus.db").stat().st_size,
                  table_page_bytes=pages, combined_db_sha256=digest(work/"corpus.db"),
                  script_sha256=digest(Path(__file__)),protocol_sha256=digest(HERE/"recorded/protocol.json"))
    (work / "baseline-manifest.json").write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['baseline_only_bytes','combined_db_bytes','baseline_build_seconds','sqlite']}))


if __name__ == "__main__":
    main()
