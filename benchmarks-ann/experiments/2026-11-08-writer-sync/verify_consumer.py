"""Compile an independent consumer against a committed writer and verified native bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def call(args, **kwargs):
    result = subprocess.run(args, text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"{args!r} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    revision = call(["git", "rev-parse", args.revision], cwd=repo).stdout.strip()
    bundle = repo / "dist/bundles/949073f-default"
    call(["python3", str(repo / "scripts/local_bundle.py"), "verify", str(bundle)])
    extension = bundle / "lib/vec0.dylib"
    extension_hash = hashlib.sha256(extension.read_bytes()).hexdigest()
    assert extension_hash == "4785d6f2ef2b1da23ed8c99f218b9dd519f4270e0789a3a20d52f8727f4c0261"
    with tempfile.TemporaryDirectory(prefix="vec-sync-consumer-") as directory:
        work = Path(directory)
        names = call(["git", "ls-tree", "-r", "--name-only", revision, "examples/write-prototype"], cwd=repo).stdout.splitlines()
        for name in names:
            relative = Path(name).relative_to("examples/write-prototype")
            target = work / "writer" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=repo))
        consumer = work / "consumer"
        (consumer / "src").mkdir(parents=True)
        (consumer / "src/main.rs").write_bytes((here / "consumer.rs").read_bytes())
        (consumer / "Cargo.toml").write_text('''[package]
name = "independent-sync-consumer"
version = "0.1.0"
edition = "2024"
[dependencies]
sqlite-vec-write-prototype = { path = "../writer" }
rusqlite = { version = "=0.40.2", features = ["bundled", "session", "load_extension"] }
''')
        env = dict(os.environ, CARGO_TARGET_DIR=str(work / "target"))
        manifest = str(consumer / "Cargo.toml")
        call(["cargo", "generate-lockfile", "--offline", "--manifest-path", manifest], env=env)
        lint = call(["cargo", "clippy", "--offline", "--locked", "--manifest-path", manifest, "--", "-D", "warnings"], env=env)
        result = call(["cargo", "run", "--offline", "--locked", "--release", "--manifest-path", manifest,
                       "--", str(work / "acceptance.db"), str(extension)], env=env)
        report = dict(revision=revision, extension_sha256=extension_hash,
                      consumer_sha256=hashlib.sha256((here / "consumer.rs").read_bytes()).hexdigest(),
                      lockfile=(consumer / "Cargo.lock").read_text(), stdout=result.stdout,
                      build_stderr=result.stderr, clippy_stderr=lint.stderr)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(result.stdout.strip())
    print(args.output)


if __name__ == "__main__":
    main()
