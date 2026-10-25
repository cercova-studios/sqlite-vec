"""Compare lance0 SQL before/after opt-in ANN; emit one JSON record.

Run from the repository root with the tests Python environment after make hybrid.
"""
import argparse
import json
import resource
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", choices=["gaussian", "clustered"], default="gaussian")
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    n, dim, k, nq = 20000, 128, 10, 100
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    queries = rng.normal(size=(nq, dim)).astype(np.float32)
    if args.distribution == "clustered":
        centers = rng.normal(size=(64, dim)).astype(np.float32) * 5
        vectors += centers[rng.integers(64, size=n)]
        queries += centers[rng.integers(64, size=nq)]
    truth = [set(np.argsort(np.sum((vectors - q) ** 2, axis=1))[:k]) for q in queries]

    with tempfile.TemporaryDirectory(prefix="lance-ann-") as root:
        uri = Path(root) / "docs.lance"
        with sqlite3.connect(Path(root) / "app.db", isolation_level=None) as db:
            db.enable_load_extension(True)
            db.load_extension("dist/vec0_lance")
            db.enable_load_extension(False)
            db.execute(f"create virtual table docs using lance0('{uri}', {dim})")
            db.execute("begin")
            db.executemany("insert into docs(rowid, vector) values (?, ?)",
                           ((i, v.tobytes()) for i, v in enumerate(vectors)))
            db.execute("commit")

            def measure():
                latencies, hits = [], 0
                sql = "select rowid, distance from docs where query = ? and k = ? order by distance"
                db.execute(sql, [queries[0].tobytes(), k]).fetchall()  # warm-up
                for q, expected in zip(queries, truth):
                    start = time.perf_counter()
                    got = db.execute(sql, [q.tobytes(), k]).fetchall()
                    latencies.append((time.perf_counter() - start) * 1000)
                    assert len(got) == k and len({i for i, _ in got}) == k
                    assert np.allclose([d for _, d in got],
                                       [np.linalg.norm(vectors[i] - q) for i, _ in got], rtol=1e-5)
                    hits += len({i for i, _ in got} & expected)
                return {"recall_at_10": hits / (nq * k),
                        "p50_ms": float(np.percentile(latencies, 50)),
                        "p95_ms": float(np.percentile(latencies, 95))}

            before = measure()
            assert before["recall_at_10"] == 1.0
            disk_before = sum(p.stat().st_size for p in uri.rglob("*") if p.is_file())
            start = time.perf_counter()
            db.execute("insert into docs(docs) values ('optimize')")
            build_seconds = time.perf_counter() - start
            after = measure()
            version = db.execute("select v from docs_lance where k='version'").fetchone()[0]
            inspect = subprocess.run(
                ["cargo", "run", "--release", "--offline", "--manifest-path", "lance/Cargo.toml",
                 "--example", "index-plan", "--", str(uri), str(version), str(dim)],
                check=True, stdout=subprocess.PIPE, text=True,
            )
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            stats_line, plan = inspect.stdout.split("\n", 1)
            stats = json.loads(stats_line)
            assert stats["index_type"] == "IVF_HNSW_SQ"
            result = {"distribution": args.distribution, "rows": n, "dimensions": dim,
                      "queries": nq, "seed": 42, "exact": before, "ann": after,
                      "optimize_seconds": build_seconds,
                      "process_peak_rss_mb": rss / (1024 ** 2 if sys.platform == "darwin" else 1024),
                      "disk_before_bytes": disk_before,
                      "disk_after_bytes": sum(p.stat().st_size for p in uri.rglob("*") if p.is_file()),
                      "index_type": stats["index_type"],
                      "partitions": stats["indices"][0]["num_partitions"],
                      "index_plan": plan}
            print(json.dumps(result), flush=True)
            assert after["recall_at_10"] >= 0.95, "ANN recall below benchmark acceptance target"


if __name__ == "__main__":
    main()
