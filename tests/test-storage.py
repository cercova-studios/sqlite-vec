"""vec0 storage providers, in a build without one (the plain C extension)."""
import sqlite3

import pytest


@pytest.fixture()
def db():
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension("dist/vec0")
    db.enable_load_extension(False)
    return db


def test_storage_provider_missing(db):
    with pytest.raises(sqlite3.OperationalError, match="storage provider 'lance' is not available.*vec0_lance"):
        db.execute("create virtual table t using vec0(v float[2], +doc blob, storage=lance, storage_uri='/tmp/x')")


def test_storage_options_validated(db):
    with pytest.raises(sqlite3.OperationalError, match="storage_uri needs storage"):
        db.execute("create virtual table t using vec0(v float[2], +doc blob, storage_uri='/tmp/x')")
    for bad in ["storage_uri=/tmp/x", "storage_uri='a' junk"]:
        with pytest.raises(sqlite3.OperationalError, match="single-quoted"):
            db.execute(f"create virtual table t using vec0(v float[2], +doc blob, storage=lance, {bad})")
    # quotes inside the string use SQL escaping
    with pytest.raises(sqlite3.OperationalError, match="not available"):
        db.execute("create virtual table t using vec0(v float[2], +doc blob, storage=lance, storage_uri='it''s')")
