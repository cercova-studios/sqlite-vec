"""Verify frozen receipts and regenerate summaries without benchmarking."""
from pathlib import Path
import ast
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ARCHIVE = Path(__file__).resolve().parent

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    for name, receipt in manifest["files"].items():
        data = (ARCHIVE / name).read_bytes()
        assert len(data) == receipt["bytes"] and sha(data) == receipt["sha256"], name
        if "uncompressed_sha256" in receipt:
            assert sha(gzip.decompress(data)) == receipt["uncompressed_sha256"], name
    for script in ARCHIVE.rglob("*.py"):
        ast.parse(script.read_text(), filename=str(script))
    harness = sha((ARCHIVE / "recorded/scripts/benchmark.py").read_bytes())
    builds = json.loads((ARCHIVE / "provenance/build-manifest.json").read_text())
    with tempfile.TemporaryDirectory(prefix="hnsw-alpha-verify-") as directory:
        work = Path(directory)
        (work / "results").mkdir()
        (work / "builds").mkdir()
        (work / "builds/manifest.json").write_text(json.dumps(builds))
        jobs = []
        for path in sorted((ARCHIVE / "results/raw").glob("*.json.gz")):
            raw = gzip.decompress(path.read_bytes())
            job = json.loads(raw)
            assert job["harness_sha256"] == harness
            assert job["rows"] == 50000 and job["queries"] == 1000
            assert job["library_sha256"] == builds["variants"][f"alpha-{job['alpha']:g}"]["library_sha256"]
            assert set(job["configurations"]) == {"32", "64", "128", "256"}
            jobs.append((job["dataset"], job["alpha"], job["seed"]))
            (work / "results" / path.stem).write_bytes(raw)
        assert len(jobs) == len(set(jobs)) == 24
        (work / "paired_timing.json").write_bytes(gzip.decompress((ARCHIVE / "results/paired_timing.json.gz").read_bytes()))
        env = dict(os.environ, HNSW_ALPHA_WORKDIR=str(work), PYTHONDONTWRITEBYTECODE="1")
        for script in ("analyze.py", "paired_analyze.py"):
            subprocess.run([sys.executable, str(ARCHIVE / "scripts" / script)], env=env, check=True, timeout=60)
        for name in ("summary.json", "paired_summary.json"):
            assert json.loads((work / name).read_text()) == json.loads((ARCHIVE / "results" / name).read_text()), name
        assert (work / "summary.csv").read_bytes() == (ARCHIVE / "results/summary.csv").read_bytes()
    print("Archive verified: hashes, provenance, syntax, 24 builds, both regenerated summaries, 640000 timed queries.")

if __name__ == "__main__":
    main()
