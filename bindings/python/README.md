# Use this fork from Python

Install the fork into an application's virtual environment. It provides
`sqlite_vec.load`, `get_loadable_path`, `serialize_float32` and `serialize_int8`,
with the fork's native vector indexes and graph SQL functions.

From a checkout of this repository:

```sh
uv pip install --python /path/to/application/.venv/bin/python .
```

Source builds need Python 3.9+, a C compiler and setuptools 77+. The build frontend
installs setuptools in its isolated build environment. The committed native inputs
include SQLite headers; builds do not need Cargo, Make or a vendor download.
Python must provide `sqlite3.Connection.enable_load_extension`; some macOS system
Python builds disable it. Use SQLite 3.41+ for the fork's bounded graph evidence
function. Check the application's runtime with `sqlite3.sqlite_version`.

```python
import sqlite3
import sqlite_vec

with sqlite3.connect(":memory:") as db:
    db.enable_load_extension(True)
    try:
        sqlite_vec.load(db)
    finally:
        db.enable_load_extension(False)

    db.execute("CREATE VIRTUAL TABLE notes USING vec0(embedding float[3])")
    db.execute("INSERT INTO notes(rowid, embedding) VALUES (?, ?)",
               (1, sqlite_vec.serialize_float32([0.1, 0.2, 0.3])))
    print(db.execute(
        "SELECT rowid FROM notes WHERE embedding MATCH ? AND k = 1",
        (sqlite_vec.serialize_float32([0.1, 0.2, 0.3]),),
    ).fetchall())  # [(1,)]
```

Load the extension separately on each connection, including connections managed
by a pool or asynchronous wrapper. The caller controls loading permissions and
transactions. The package uses the application's SQLite runtime and adds no
runtime Python dependencies.

## Declare an application dependency

For sibling checkouts such as `obsidian-agent` and `sqlite-vec`, run this from the
application directory:

```sh
uv add ../sqlite-vec
```

This records a local source dependency in the application. For a portable Git
dependency after publishing the package changes, pin a reviewed fork revision:

```sh
uv add 'sqlite-vec @ git+https://github.com/cercova-studios/sqlite-vec.git@<full-commit-sha>'
```

The distribution keeps the `sqlite-vec` name and `sqlite_vec` import. Its version
is root `VERSION`, normalized to Python packaging syntax, plus `+cercova.1`.
An unqualified `pip install sqlite-vec` selects the upstream registry package;
use an explicit local, wheel or Git source for fork features. Install one provider
of `sqlite_vec` per environment.

## Install a released build

Releases publish wheels and a source archive to a public Google Artifact Registry
index. Installing needs no credentials. Route only this package to that index:

```toml
[project]
dependencies = ["sqlite-vec>=0.1.10a4"]

[[tool.uv.index]]
name = "cercova"
url = "https://us-east4-python.pkg.dev/gen-lang-client-0026636413/sqlite-vec-python/simple/"
explicit = true

[tool.uv.sources]
sqlite-vec = { index = "cercova" }
```

`explicit = true` keeps every other dependency on PyPI, so the upstream
`sqlite-vec` there is never selected by accident. uv installs a pre-release only
when the requirement names one, as `>=0.1.10a4` does. Wheels cover manylinux
x86_64/aarch64, macOS x86_64/arm64 and Windows x86_64; other platforms build the
source archive and need a C compiler. The same files are attached to each
[GitHub release](https://github.com/cercova-studios/sqlite-vec/releases).

## Build a wheel or source archive

From this repository's root:

```sh
uv build --out-dir dist/python
uv pip install --python /path/to/application/.venv/bin/python dist/python/*.whl
```

The wheel contains a platform-specific SQLite loadable library, with no dependency
on the Python C ABI. It can serve compatible Python versions on the same platform.
Build on the deployment platform; local Linux wheels are not automatically
manylinux releases. Default builds use portable scalar kernels and include native
HNSW, DiskANN, sparse vectors and graph traversal. Lance and experimental IVF are
separate build profiles and are not included.

## Maintain and verify

Python and Rust share the committed C amalgamation and headers under
`bindings/rust`. After native changes, follow the
[Rust input refresh procedure](../rust/README.md#maintain-the-package) and commit
the refreshed inputs with the change. Consumer builds compile those inputs
without regenerating them.

Run the packaging checks with Python 3.11+, uv and setuptools 77+ installed in
that interpreter. The checks run offline without build isolation:

```sh
python3 tests/python-packaging.py -v
```

For example, use `uv run --no-project --with 'setuptools>=77' python
tests/python-packaging.py -v` to supply setuptools in a separate environment.

The smoke check builds a source archive and wheel outside the checkout, validates
the wheel's platform/ABI tags, and installs the wheel into a fresh environment.
It loads the installed extension and runs one HNSW query to confirm fork features
are present. Shared-source freshness is checked by `tests/rust-packaging.py`;
native regressions cover transaction and graph behavior. Use a disposable database
to qualify an existing application's schema before switching writers; see
[database compatibility](../rust/README.md#database-compatibility).
