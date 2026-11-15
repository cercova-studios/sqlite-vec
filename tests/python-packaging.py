"""Build and exercise isolated Python consumers using uv and installed setuptools."""

from pathlib import Path
import json
import os
import shutil
import string
import subprocess
import sys
import tarfile
import tempfile
import unittest
import venv
import zipfile

REPO = Path(__file__).resolve().parents[1]
PROBE = r"""
from pathlib import Path
from importlib.metadata import version
import json
import sqlite3
import struct
import sys
import sqlite_vec

assert Path(sqlite_vec.__file__).is_relative_to(Path(sys.prefix))
assert version("sqlite-vec").endswith("+cercova.1")
assert sqlite_vec.serialize_float32([1, 2]) == struct.pack("2f", 1, 2)
assert sqlite_vec.serialize_int8([-128, 127]) == b"\x80\x7f"
try:
    sqlite_vec.serialize_int8([128])
except struct.error:
    pass
else:
    raise AssertionError("int8 overflow was accepted")

def connect():
    db = sqlite3.connect("consumer.db")
    db.enable_load_extension(True)
    try:
        sqlite_vec.load(db)
    finally:
        db.enable_load_extension(False)
    assert db.execute("SELECT vec_version()").fetchone() == (sys.argv[1],)
    try:
        sqlite_vec.load(db)
    except sqlite3.OperationalError as error:
        assert "not authorized" in str(error), str(error)
    else:
        raise AssertionError("extension loading remained enabled")
    return db

db = connect()
db.execute("CREATE VIRTUAL TABLE flat USING vec0(v float[2])")
db.execute("CREATE VIRTUAL TABLE ann USING vec0(v float[2] indexed by hnsw(m=4))")
for table in ("flat", "ann"):
    db.executemany(f"INSERT INTO {table}(rowid,v) VALUES (?,?)",
                   [(1, sqlite_vec.serialize_float32([0,0])),
                    (2, sqlite_vec.serialize_float32([3,4]))])
db.commit()
for table in ("flat", "ann"):
    assert db.execute(f"SELECT rowid,distance FROM {table} "
                      "WHERE v MATCH '[0,0]' AND k=2").fetchall() == [(1,0.0),(2,5.0)]
    db.execute("BEGIN")
    db.execute(f"INSERT INTO {table}(rowid,v) VALUES (3,'[0,1]')")
    db.execute("SAVEPOINT s")
    db.execute(f"DELETE FROM {table} WHERE rowid=1")
    assert db.execute(f"SELECT rowid FROM {table} WHERE v MATCH '[0,0]' AND k=1").fetchall() == [(3,)]
    db.execute("ROLLBACK TO s")
    db.execute("RELEASE s")
    assert db.execute(f"SELECT rowid FROM {table} WHERE v MATCH '[0,0]' AND k=1").fetchall() == [(1,)]
    db.rollback()
    assert db.execute(f"SELECT rowid FROM {table} ORDER BY rowid").fetchall() == [(1,),(2,)]

db.execute("CREATE TABLE edges(id INTEGER PRIMARY KEY, src INTEGER, dst INTEGER)")
db.executemany("INSERT INTO edges VALUES (?,?,?)", [(7,1,2),(8,2,3)])
assert dict(db.execute("SELECT node,depth FROM graph_bfs('edges','src','dst',1)")) == {1:0,2:1,3:2}
assert json.loads(db.execute("SELECT graph_witness('edges','src','dst','id',1,3,1000,65536)").fetchone()[0]) == [[1,None],[2,7],[3,8]]
db.execute("CREATE VIRTUAL TABLE lexical USING fts5(text)")
db.execute("INSERT INTO lexical VALUES ('obsidian note')")
assert db.execute("SELECT rowid FROM lexical WHERE lexical MATCH 'obsidian'").fetchall() == [(1,)]
db.commit()
db.close()
db = connect()
for table in ("flat", "ann"):
    assert db.execute(f"SELECT rowid,distance FROM {table} "
                      "WHERE v MATCH '[3,4]' AND k=1").fetchall() == [(2,0.0)]
assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
db.close()
print(json.dumps({"python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version,
                  "package": version("sqlite-vec"), "checks": "passed"}))
"""


def run(command, cwd, env=None):
    result = subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True, timeout=300
    )
    if result.returncode:
        raise AssertionError(f"{command}\n{result.stdout}\n{result.stderr}")
    return result.stdout


class PythonPackaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="sqlite-vec-python-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "source"
        names = [
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
        ]
        for name in names:
            target = cls.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, target)
        cls.env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("PYTHONPATH", "PYTHONHOME")
        }
        cls.env["UV_CACHE_DIR"] = str(cls.root / "cache")
        cls.output = cls.root / "artifacts"
        run(
            [
                "uv",
                "build",
                "--offline",
                "--no-build-isolation",
                "--python",
                sys.executable,
                "--out-dir",
                str(cls.output),
            ],
            cls.source,
            cls.env,
        )
        (cls.wheel,) = cls.output.glob("*.whl")
        (cls.sdist,) = cls.output.glob("*.tar.gz")

    def consume(self, wheel, name, source=False):
        consumer = self.root / name
        consumer.mkdir()
        environment = consumer / "venv"
        venv.EnvBuilder(with_pip=False).create(environment)
        python = environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        if source:
            site = run(
                [
                    str(python),
                    "-c",
                    "import sysconfig; print(sysconfig.get_path('platlib'))",
                ],
                consumer,
                self.env,
            ).strip()
            run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--offline",
                    "--no-deps",
                    "--no-build-isolation",
                    "--python",
                    sys.executable,
                    "--target",
                    site,
                    str(wheel),
                ],
                consumer,
                self.env,
            )
        else:
            run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--offline",
                    "--no-deps",
                    "--python",
                    str(python),
                    str(wheel),
                ],
                consumer,
                self.env,
            )
        result = run(
            [str(python), "-c", PROBE, "v" + (REPO / "VERSION").read_text().strip()],
            consumer,
            self.env,
        )
        self.assertEqual(json.loads(result)["checks"], "passed")

    def test_local_source_install(self):
        self.consume(self.source, "local-consumer", source=True)

    def test_shared_inputs_match_current_sources(self):
        source = run([sys.executable, "scripts/amalgamate.py", "sqlite-vec.c"], REPO)
        self.assertEqual(
            source, (self.source / "bindings/rust/sqlite-vec.c").read_text()
        )
        version = (REPO / "VERSION").read_text().strip()
        header = (self.source / "bindings/rust/sqlite-vec.h").read_text()
        values = dict(
            VERSION=version,
            VERSION_MAJOR=version.split(".")[0],
            VERSION_MINOR=version.split(".")[1],
            VERSION_PATCH=version.split(".")[2].split("-")[0],
        )
        for macro in ("DATE", "SOURCE"):
            line = next(
                line
                for line in header.splitlines()
                if line.startswith(f"#define SQLITE_VEC_{macro} ")
            )
            values[macro] = line.split('"')[1]
        self.assertEqual(
            header,
            string.Template((REPO / "sqlite-vec.h.tmpl").read_text()).substitute(
                values
            ),
        )

    def test_wheel_contents_and_clean_consumer(self):
        self.assertIn("-py3-none-", self.wheel.name)
        self.assertNotIn("-any.whl", self.wheel.name)
        with zipfile.ZipFile(self.wheel) as archive:
            names = archive.namelist()
            self.assertTrue(any(name.startswith("sqlite_vec/vec0.") for name in names))
            metadata = archive.read(
                next(name for name in names if name.endswith("/METADATA"))
            ).decode()
            self.assertNotIn("Requires-Dist:", metadata)
            self.assertIn("License-Expression: MIT OR Apache-2.0", metadata)
            for name in ("LICENSE-MIT", "LICENSE-APACHE"):
                self.assertTrue(
                    any(path.endswith("/licenses/" + name) for path in names)
                )
        self.consume(self.wheel, "wheel-consumer")

    def test_sdist_rebuild_without_checkout(self):
        # The default uv build already builds its wheel from this source archive.
        with tarfile.open(self.sdist) as archive:
            names = archive.getnames()
            for name in ("sqlite-vec.c", "sqlite-vec.h", "sqlite3.h", "sqlite3ext.h"):
                self.assertTrue(
                    any(path.endswith("/bindings/rust/" + name) for path in names)
                )
        relocated = self.root / "relocated.tar.gz"
        shutil.copyfile(self.sdist, relocated)
        hidden = self.root / "hidden-source"
        self.source.rename(hidden)
        try:
            output = self.root / "rebuilt"
            run(
                [
                    "uv",
                    "build",
                    "--offline",
                    "--no-build-isolation",
                    "--python",
                    sys.executable,
                    "--wheel",
                    str(relocated),
                    "--out-dir",
                    str(output),
                ],
                self.root,
                self.env,
            )
            (wheel,) = output.glob("*.whl")
            self.consume(wheel, "sdist-consumer")
        finally:
            hidden.rename(self.source)


if __name__ == "__main__":
    unittest.main()
