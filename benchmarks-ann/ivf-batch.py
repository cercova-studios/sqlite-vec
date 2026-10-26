"""Compare IVF write batching with the pre-change experimental-IVF build.

Run from the repository root. Builds run in separate processes and temporary
databases. The oracle ranks original float32 vectors independently in Python.
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


def run(extension, path, rows, trained, quantizer):
    rng = random.Random(26)
    values = [struct.unpack("32f", struct.pack("32f", *(rng.gauss(0, 1) for _ in range(32))))
              for _ in range(rows + 512)]
    blobs = [struct.pack("32f", *v) for v in values]
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    db.load_extension(str(extension))
    db.enable_load_extension(False)
    assert "ivf" in db.execute("select vec_debug()").fetchone()[0].split("Build flags:")[-1]
    db.execute("pragma journal_mode=wal")
    db.execute("pragma wal_autocheckpoint=0")
    extra = "" if quantizer == "none" else ",oversample=64"
    db.execute(f"create virtual table t using vec0(v float[32] indexed by "
               f"ivf(nlist=16,nprobe=16,quantizer={quantizer}{extra}))")
    db.executemany("insert into t(rowid,v) values (?,?)", enumerate(blobs[:512]))
    if trained:
        db.execute("insert into t(t) values ('compute-centroids')")
    db.commit()
    db.execute("pragma wal_checkpoint(truncate)")
    changes = db.total_changes
    start = time.perf_counter()
    db.executemany("insert into t(rowid,v) values (?,?)", enumerate(blobs[512:], start=512))
    insert_seconds = time.perf_counter() - start
    start = time.perf_counter()
    db.commit()
    commit_seconds = time.perf_counter() - start
    changes = db.total_changes - changes
    wal_bytes = Path(str(path) + "-wal").stat().st_size
    assert db.execute("select sum(n_vectors) from t_ivf_cells00").fetchone()[0] == len(blobs)
    timings, answers, hits = [], [], 0
    for q in values[512:522]:
        expected = sorted(range(len(values)), key=lambda i: sum((x-y)**2 for x,y in zip(values[i], q)))[:10]
        start = time.perf_counter()
        result = db.execute("select rowid,distance from t where v match ? and k=10", [struct.pack("32f", *q)]).fetchall()
        timings.append(time.perf_counter() - start)
        answers.append(result)
        hits += len(set(expected) & {r[0] for r in result})
        for i, distance in result:
            want = sum((x-y)**2 for x,y in zip(values[i], q)) ** .5
            assert abs(want - distance) <= 1e-5 * max(1., want)
    assert hits >= 95
    db.close()
    return {"insert_seconds": insert_seconds, "commit_seconds": commit_seconds,
            "write_seconds": insert_seconds + commit_seconds, "shadow_changes": changes,
            "wal_bytes": wal_bytes, "query_mean_ms": sum(timings) * 1000 / len(timings),
            "recall_at_10": hits / 100, "answers": answers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=Path("dist/vec0"))
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--trained", action="store_true")
    parser.add_argument("--quantizer", choices=["none", "int8"], default="none")
    args = parser.parse_args()
    if args.rows < 10:
        parser.error("rows must be >= 10")
    with tempfile.TemporaryDirectory(prefix="vec-ivf-batch-") as root:
        with multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=1) as worker:
            result = {name: worker.apply(run, (extension.resolve(), Path(root) / f"{name}.db",
                                               args.rows, args.trained, args.quantizer))
                      for name, extension in [("baseline", args.baseline), ("candidate", args.candidate)]}
        assert result["candidate"].pop("answers") == result["baseline"].pop("answers")
        assert result["candidate"]["shadow_changes"] < result["baseline"]["shadow_changes"]
        print(json.dumps({"rows_inserted": args.rows, "dimensions": 32,
                          "trained": args.trained, "quantizer": args.quantizer, **result}))


if __name__ == "__main__":
    main()
