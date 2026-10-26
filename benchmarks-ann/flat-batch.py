"""Compare flat-vector batching against the previous default extension build.

Run from the repository root. Isolated processes prevent extension-symbol
interactions; temporary WAL databases use the same deterministic float32 data.
"""
import argparse
import json
import multiprocessing
import random
import sqlite3
import struct
import tempfile
import time
from pathlib import Path


def run(extension, path, rows, dimensions):
    rng = random.Random(26)
    blobs = [struct.pack(f"{dimensions}f", *(rng.gauss(0, 1) for _ in range(dimensions))) for _ in range(rows)]
    values = [struct.unpack(f"{dimensions}f", v) for v in blobs]
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    db.load_extension(str(extension))
    db.execute("pragma journal_mode=wal")
    db.execute("pragma wal_autocheckpoint=0")
    db.execute(f"create virtual table t using vec0(v float[{dimensions}],chunk_size=1024)")
    db.commit()
    db.execute("pragma wal_checkpoint(truncate)")
    changes = db.total_changes
    start = time.perf_counter()
    db.executemany("insert into t(rowid,v) values (?,?)", enumerate(blobs))
    insert_seconds = time.perf_counter() - start
    start = time.perf_counter()
    db.commit()
    commit_seconds = time.perf_counter() - start
    changes = db.total_changes - changes
    wal_bytes = Path(str(path) + "-wal").stat().st_size
    answers, times = [], []
    for q in values[:10]:
        want = sorted(range(rows), key=lambda i: sum((x-y)**2 for x,y in zip(values[i], q)))[:10]
        start = time.perf_counter()
        result = db.execute("select rowid,distance from t where v match ? and k=10", [struct.pack(f"{dimensions}f", *q)]).fetchall()
        times.append(time.perf_counter() - start)
        assert [r[0] for r in result] == want
        for i, distance in result:
            expected = sum((x-y)**2 for x,y in zip(values[i], q)) ** .5
            assert abs(distance - expected) < 1e-5 * max(1., expected)
        answers.append(result)
    db.close()
    return {"write_seconds": insert_seconds + commit_seconds,
            "insert_seconds": insert_seconds, "commit_seconds": commit_seconds,
            "shadow_changes": changes, "wal_bytes": wal_bytes,
            "query_mean_ms": sum(times) * 1000 / len(times), "answers": answers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=Path("dist/vec0"))
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dimensions", type=int, default=128)
    args = parser.parse_args()
    if args.rows < 10 or args.dimensions < 1:
        parser.error("require rows >= 10 and dimensions >= 1")
    with tempfile.TemporaryDirectory(prefix="vec-flat-batch-") as root:
        with multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=1) as worker:
            result = {name: worker.apply(run, (extension.resolve(), Path(root) / f"{name}.db", args.rows, args.dimensions))
                      for name, extension in [("baseline", args.baseline), ("candidate", args.candidate)]}
        assert result["candidate"].pop("answers") == result["baseline"].pop("answers")
        print(json.dumps({"rows": args.rows, "dimensions": args.dimensions, **result}))


if __name__ == "__main__":
    main()
