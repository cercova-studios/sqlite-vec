"""Check committed Cargo inputs, a pinned Git consumer and 0.1.9 file interchange.

Run with Python 3.11+, Cargo and a C compiler. Registry inputs must be cached;
the only Cargo fetch uses a disposable local Git repository.
"""
from pathlib import Path
import os
import shutil
import string
import subprocess
import tempfile
import tomllib
import unittest

REPO = Path(__file__).resolve().parents[1]


def run(command, cwd, **kwargs):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                            timeout=300, **kwargs)
    if result.returncode:
        raise AssertionError(f"{command}\n{result.stdout}\n{result.stderr}")
    return result.stdout


class RustPackaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="sqlite-vec-rust-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cargo_home = cls.root / "cargo"
        cargo_home.mkdir()
        registry = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")) / "registry"
        (cargo_home / "registry").symlink_to(registry, target_is_directory=True)
        cls.env = dict(os.environ, CARGO_HOME=str(cargo_home), CARGO_NET_GIT_FETCH_WITH_CLI="true")
        cls.package = cls.root / "fork" / "bindings" / "rust"
        cls.package.parent.mkdir(parents=True)
        shutil.copytree(REPO / "bindings/rust", cls.package, ignore=shutil.ignore_patterns("target"))
        git = cls.root / "fork"
        run(["git", "init", "--quiet"], git)
        run(["git", "add", "bindings/rust"], git)
        run(["git", "-c", "user.name=Packaging test", "-c", "user.email=test@example.invalid",
             "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "-m", "test: package snapshot"], git)
        cls.revision = run(["git", "rev-parse", "HEAD"], git).strip()
        cls.version = tomllib.loads((cls.package / "Cargo.toml").read_text())["package"]["version"]

    def test_generated_inputs_match_sources(self):
        source = run(["python3", "scripts/amalgamate.py", "sqlite-vec.c"], REPO)
        self.assertEqual(source, (self.package / "sqlite-vec.c").read_text())
        self.assertEqual(self.version, (REPO / "VERSION").read_text().strip())
        header = (self.package / "sqlite-vec.h").read_text()
        values = dict(VERSION=self.version, VERSION_MAJOR=self.version.split(".")[0],
                      VERSION_MINOR=self.version.split(".")[1],
                      VERSION_PATCH=self.version.split(".")[2].split("-")[0])
        for macro in ("DATE", "SOURCE"):
            line = next(line for line in header.splitlines() if line.startswith(f"#define SQLITE_VEC_{macro} "))
            values[macro] = line.split('"')[1]
        self.assertEqual(header, string.Template((REPO / "sqlite-vec.h.tmpl").read_text()).substitute(values))
        for name in ("sqlite3.h", "sqlite3ext.h"):
            self.assertEqual((self.package / name).read_bytes(), (REPO / "vendor" / name).read_bytes())

    def test_bundled_registration_and_crate_archive(self):
        run(["cargo", "test", "--offline", "--locked", "--features", "rusqlite/bundled"], self.package, env=self.env)
        # Cargo builds the unpacked archive too: parent sources cannot rescue missing inputs.
        run(["cargo", "package", "--offline", "--locked"], self.package, env=self.env)

    def test_pinned_git_and_019_compatibility(self):
        legacy = self.root / "legacy"
        current = self.root / "current"
        for folder in (legacy, current):
            shutil.copytree(REPO / "tests/rust-compat", folder, ignore=shutil.ignore_patterns("target"))
        manifest = current / "Cargo.toml"
        dependency = f'sqlite-vec = {{ git = "{(self.root / "fork").as_uri()}", rev = "{self.revision}" }}'
        manifest.write_text(manifest.read_text().replace('sqlite-vec = "=0.1.9"', dependency))
        run(["cargo", "build", "--offline", "--locked"], legacy, env=self.env)
        # Reuse the locked registry graph, replacing only the extension with the Git pin.
        lock = current / "Cargo.lock"
        blocks = lock.read_text().split("\n[[package]]\n")
        for i, block in enumerate(blocks[1:], 1):
            if tomllib.loads(block)["name"] == "sqlite-vec":
                blocks[i] = (f'name = "sqlite-vec"\nversion = "{self.version}"\n'
                             f'source = "git+{(self.root / "fork").as_uri()}?rev={self.revision}#{self.revision}"\n'
                             'dependencies = [\n "cc",\n]\n')
        lock.write_text("\n[[package]]\n".join(blocks))
        run(["cargo", "build", "--locked", "--config", "net.retry=0"], current, env=self.env)
        old = legacy / "target/debug/sqlite-vec-compat"
        new = current / "target/debug/sqlite-vec-compat"

        def check(binary, operation, path):
            version = "v0.1.9" if binary == old else "v" + self.version
            run([str(binary), operation, str(path), version], self.root)

        path = self.root / "old.db"
        check(old, "create", path)
        check(new, "read", path)
        check(new, "write", path)
        check(new, "read", path)
        check(old, "read", path)
        check(old, "write", path)
        check(new, "read", path)
        check(new, "mixed-chunks", path)
        check(new, "verify-mixed", path)
        check(old, "verify-mixed", path)
        path = self.root / "new.db"
        check(new, "create", path)
        check(new, "write", path)
        check(old, "read", path)
        # Downgrade writes must invalidate persisted bounds before the old writer runs.
        check(new, "clear-zonemaps", path)
        check(old, "legacy-write", path)
        check(new, "verify-legacy-write", path)
        check(new, "features", self.root / "features.db")


if __name__ == "__main__":
    unittest.main()
