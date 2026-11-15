"""Package the shared C inputs as a SQLite extension, without the Python C API."""

from pathlib import Path
import os
import sys

from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext


class SQLiteBuild(build_ext):
    def get_ext_filename(self, fullname):
        suffix = (
            ".dll"
            if os.name == "nt"
            else ".dylib"
            if sys.platform == "darwin"
            else ".so"
        )
        return os.path.join(*fullname.split(".")) + suffix

    def get_export_symbols(self, ext):
        # SQLite loads this library; there is no Python module initializer.
        return ["sqlite3_vec_init"]


class SQLiteWheel(bdist_wheel):
    def get_tag(self):
        return "py3", "none", super().get_tag()[2]


setup(
    version=Path("VERSION").read_text().strip() + "+cercova.1",
    ext_modules=[
        Extension(
            "sqlite_vec.vec0",
            sources=["bindings/rust/sqlite-vec.c"],
            include_dirs=["bindings/rust"],
            depends=[
                "bindings/rust/sqlite-vec.h",
                "bindings/rust/sqlite3.h",
                "bindings/rust/sqlite3ext.h",
            ],
            libraries=[] if os.name == "nt" else ["m"],
            extra_compile_args=["/O2"] if os.name == "nt" else ["-O3", "-std=c11"],
        )
    ],
    cmdclass={"build_ext": SQLiteBuild, "bdist_wheel": SQLiteWheel},
)
