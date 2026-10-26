"""Compare HNSW transaction writes and time against a previous extension build.

Run from the repo root. Both builds use identical deterministic vectors, graph
parameters and queries, in separate processes to isolate exported C symbols.
"""
import argparse
import hashlib
import json
import multiprocessing
import random
import sqlite3
import struct
import tempfile
import time
from pathlib import Path


def run(extension, path, rows, dimensions, cache_mb):
    rng = random.Random(17)
    vectors = [struct.pack(f"{dimensions}f", *(rng.gauss(0, 1) for _ in range(dimensions)))
               for _ in range(rows)]
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    db.load_extension(str(extension))
    db.enable_load_extension(False)
    db.execute("pragma journal_mode=wal")
    db.execute("pragma wal_autocheckpoint=0")
    db.execute(f"create virtual table t using vec0(v float[{dimensions}] "
               f"indexed by hnsw(m=16,ef_construction=100,ef_search=100,cache_mb={cache_mb}))")
    db.commit()
    db.execute("pragma wal_checkpoint(truncate)")
    initial_changes = db.total_changes
    start = time.perf_counter()
    db.executemany("insert into t(rowid,v) values (?,?)", enumerate(vectors))
    insert_seconds = time.perf_counter() - start
    pending_edges = db.execute("select count(*) from t_hnsw_edges00").fetchone()[0]
    queries = [vectors[i] for i in range(0, rows, max(1, rows // 20))]
    sql = "select rowid,distance from t where v match ? and k=10"
    before = [db.execute(sql, [q]).fetchall() for q in queries]
    start = time.perf_counter()
    db.commit()
    commit_seconds = time.perf_counter() - start
    changes = db.total_changes - initial_changes
    wal_bytes = Path(str(path) + "-wal").stat().st_size
    graph = hashlib.sha256()
    for rowid, level, neighbors in db.execute("select * from t_hnsw_edges00 order by rowid,level"):
        graph.update(struct.pack("qqq", rowid, level, len(neighbors)))
        graph.update(neighbors)
    # A fresh cache must produce the same answers as the pending graph.
    db.close()
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    db.load_extension(str(extension))
    after = [db.execute(sql, [q]).fetchall() for q in queries]
    assert before == after
    assert all(result and result[0][1] == 0 for result in after)
    db.close()
    return {"insert_seconds": insert_seconds, "commit_seconds": commit_seconds,
            "write_seconds": insert_seconds + commit_seconds, "shadow_changes": changes,
            "wal_bytes": wal_bytes, "edges_before_commit": pending_edges,
            "graph_sha256": graph.hexdigest(), "results": after}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--cache-mb", type=int, default=256)
    args = parser.parse_args()
    if args.rows < 20 or not 1 <= args.dimensions <= 8192 or args.cache_mb < 0:
        parser.error("require rows >= 20, dimensions in 1..8192, cache-mb >= 0")
    with tempfile.TemporaryDirectory(prefix="vec-hnsw-batch-") as root:
        with multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=1) as worker:
            result = {}
            for name, extension in [("baseline", args.baseline), ("candidate", Path("dist/vec0"))]:
                result[name] = worker.apply(run, (extension.resolve(), Path(root) / f"{name}.db",
                                                  args.rows, args.dimensions, args.cache_mb))
        assert result["candidate"]["graph_sha256"] == result["baseline"]["graph_sha256"]
        assert result["candidate"].pop("results") == result["baseline"].pop("results")
        if args.cache_mb >= 256:
            assert result["candidate"]["shadow_changes"] < result["baseline"]["shadow_changes"]
        print(json.dumps({"rows": args.rows, "dimensions": args.dimensions,
                          "cache_mb": args.cache_mb, **result}))


if __name__ == "__main__":
    main()
