"""Measure the committed task writer through its public API on fresh fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


def call(args, **kwargs):
    result = subprocess.run(args, text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"{args!r} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="8f0af1b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    revision = call(["git", "rev-parse", args.revision], cwd=repo).stdout.strip()
    records = []
    inputs = {}
    with tempfile.TemporaryDirectory(prefix="vec-writer-measure-") as directory:
        work = Path(directory)
        source = work / "writer"
        for relative in ["Cargo.toml", "Cargo.lock", "README.md", "src/lib.rs", "src/tests.rs"]:
            data = subprocess.run(["git", "show", f"{revision}:examples/write-prototype/{relative}"], cwd=repo, check=True, capture_output=True).stdout
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            inputs[relative] = hashlib.sha256(data).hexdigest()
        bench = work / "bench"
        (bench / "src").mkdir(parents=True)
        (bench / "src/main.rs").write_bytes((here / "main.rs").read_bytes())
        (bench / "Cargo.toml").write_text('''[package]
name = "writer-path-measurement"
version = "0.1.0"
edition = "2024"
[dependencies]
sqlite-vec-write-prototype = { path = "../writer" }
rusqlite = { version = "=0.40.2", features = ["bundled", "session"] }
serde_json = "1"
''')
        env = dict(os.environ, CARGO_TARGET_DIR=str(work / "target"))
        (bench / "Cargo.lock").write_bytes((source / "Cargo.lock").read_bytes())
        call(["cargo", "generate-lockfile", "--offline", "--manifest-path", str(bench / "Cargo.toml")], env=env)
        build = call(["cargo", "build", "--offline", "--locked", "--release", "--manifest-path", str(bench / "Cargo.toml")], env=env)
        binary = work / "target/release/writer-path-measurement"
        for repeat in range(3):
            cases = [(1, 0), (16, 0), (64, 0), (16, 512), (64, 128)]
            if repeat % 2:
                cases.reverse()
            for batch, payload in cases:
                fixture = work / f"case-{repeat}-{batch}-{payload}.db"
                # One Rust child per wrapper makes RUSAGE_CHILDREN specific to this case.
                wrapper = """import json,resource,subprocess,sys
r=subprocess.run(sys.argv[1:],capture_output=True,text=True)
if r.returncode:
 print(r.stderr,file=sys.stderr)
 sys.exit(r.returncode)
record=json.loads(r.stdout)
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
record.update(peak_rss_bytes=usage.ru_maxrss,child_user_seconds=usage.ru_utime,child_system_seconds=usage.ru_stime)
print(json.dumps(record))
"""
                result = call([sys.executable, "-c", wrapper, str(binary), str(fixture), str(batch), str(payload), "50"])
                record = json.loads(result.stdout)
                record.update(repeat=repeat)
                records.append(record)
        report = dict(revision=revision, platform=platform.platform(), machine=platform.machine(),
                      rustc=call(["rustc", "--version"]).stdout.strip(),
                      input_sha256=inputs, harness_sha256=hashlib.sha256((here / "main.rs").read_bytes()).hexdigest(),
                      measurement_lockfile=(bench / "Cargo.lock").read_text(),
                      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      build_stderr=build.stderr, records=records,
                      limitations=["macOS process peak RSS includes runtime, SQLite, driver and oracle; not an operation memory bound.",
                                   "Fresh process and database per case; filesystem caches are not flushed.",
                                   "Sequential single-writer WAL/FULL workload; no model, FTS, vector, network or transport cost.",
                                   "First operation retained; no discarded warmup. Timing excludes independent SQL oracles.",
                                   "History-growth probes add 1000 committed requests after timed mutation rounds.",
                                   "Measurements are descriptive; no optimization or production latency target is asserted."])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
