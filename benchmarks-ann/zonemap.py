"""Measure ordered numeric filtering, with/without persisted chunk bounds.

Run from the repo root after make loadable. An optional old extension measures
insert overhead; each extension uses its own process and temporary database.
"""
import argparse
import json
import multiprocessing
import sqlite3
import statistics
import struct
import tempfile
import time
from pathlib import Path


def run(extension, path):
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    db.load_extension(str(extension))
    db.enable_load_extension(False)
    db.execute("create virtual table t using vec0(v float[32], score integer, chunk_size=1024)")
    start = time.perf_counter()
    db.executemany("insert into t(rowid,v,score) values (?,?,?)",
                   ((i, struct.pack("32f", float(i), *([0.] * 31)), i // 1024)
                    for i in range(65536)))
    db.commit()
    insert_seconds = time.perf_counter() - start

    def measure():
        timings = []
        sql = "select rowid from t where v match ? and k=10 and score=?"
        q = struct.pack("32f", *([0.] * 32))
        for score in range(64):
            start = time.perf_counter()
            got = [r[0] for r in db.execute(sql, [q, score])]
            timings.append((time.perf_counter() - start) * 1000)
            assert got == list(range(score * 1024, score * 1024 + 10))
        return {"p50_ms": statistics.median(timings), "p95_ms": sorted(timings)[60]}

    measure()  # warm cache
    with_bounds = measure()
    entries, payload = db.execute(
        "select count(*), coalesce(sum(length(value)),0) from t_info where key glob 'zonemap_*'"
    ).fetchone()
    db.execute("delete from t_info where key glob 'zonemap_*'")
    db.commit()
    without_bounds = measure()
    db.close()
    return {"insert_seconds": insert_seconds, "with_bounds": with_bounds,
            "without_bounds": without_bounds, "stats_entries": entries,
            "stats_payload_bytes": payload}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="previous build's loadable extension")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="vec-zonemap-") as root:
        # Keep two builds' exported C symbols out of the same process.
        with multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=1) as worker:
            result = {"rows": 65536, "dimensions": 32, "chunks": 64,
                      "candidate": worker.apply(run, (Path("dist/vec0"), Path(root) / "candidate.db"))}
            assert result["candidate"]["stats_entries"] == 64
            if args.baseline:
                result["baseline"] = worker.apply(run, (args.baseline, Path(root) / "baseline.db"))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
