"""Build a source archive and wheel, then load the installed fork in isolation."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv

REPO = Path(__file__).resolve().parents[1]
PROBE = """
from pathlib import Path
import sqlite3
import sys
import sqlite_vec

assert Path(sqlite_vec.__file__).is_relative_to(Path(sys.prefix))
with sqlite3.connect(":memory:") as db:
    db.enable_load_extension(True)
    try:
        sqlite_vec.load(db)
    finally:
        db.enable_load_extension(False)
    db.execute("CREATE VIRTUAL TABLE notes USING vec0(v float[2] indexed by hnsw(m=4))")
    db.execute("INSERT INTO notes(rowid,v) VALUES (1,?)",
               (sqlite_vec.serialize_float32([0,1]),))
    assert db.execute("SELECT rowid,distance FROM notes WHERE v MATCH '[0,1]' AND k=1").fetchall() == [(1,0.0)]
"""


def run(*command, cwd, env):
    subprocess.run(command, cwd=cwd, env=env, check=True, timeout=300)


class PythonPackaging(unittest.TestCase):
    def test_installed_wheel_loads_fork(self):
        with tempfile.TemporaryDirectory(prefix="sqlite-vec-python-") as folder:
            root = Path(folder)
            source = root / "source"
            for name in (
                "pyproject.toml",
                "setup.py",
                "MANIFEST.in",
                "VERSION",
                "README.md",
                "LICENSE-MIT",
                "LICENSE-APACHE",
                "bindings/python/__init__.py",
                "bindings/python/extra_init.py",
                "bindings/python/README.md",
                "bindings/rust/sqlite-vec.c",
                "bindings/rust/sqlite-vec.h",
                "bindings/rust/sqlite3.h",
                "bindings/rust/sqlite3ext.h",
            ):
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / name, target)
            env = {
                key: value
                for key, value in os.environ.items()
                if key not in ("PYTHONPATH", "PYTHONHOME")
            }
            env["UV_CACHE_DIR"] = str(root / "cache")
            artifacts = root / "artifacts"
            # uv builds the wheel from the source archive in this single invocation.
            run(
                "uv",
                "build",
                "--offline",
                "--no-build-isolation",
                "--python",
                sys.executable,
                "--out-dir",
                str(artifacts),
                cwd=source,
                env=env,
            )
            (wheel,) = artifacts.glob("*.whl")
            _, python_tag, abi_tag, platform_tag = wheel.stem.rsplit("-", 3)
            self.assertEqual((python_tag, abi_tag), ("py3", "none"))
            self.assertNotEqual(platform_tag, "any")

            environment = root / "venv"
            venv.EnvBuilder(with_pip=False).create(environment)
            python = environment / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            run(
                "uv",
                "pip",
                "install",
                "--offline",
                "--no-deps",
                "--python",
                str(python),
                str(wheel),
                cwd=root,
                env=env,
            )
            run(str(python), "-I", "-c", PROBE, cwd=root, env=env)


if __name__ == "__main__":
    unittest.main()
