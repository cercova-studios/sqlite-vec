import random
import sqlite3

import pytest
from helpers import _f32


@pytest.fixture()
def db():
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension("dist/vec0")
    db.enable_load_extension(False)
    return db


@pytest.mark.parametrize("elem", ["float", "int8"])
def test_vec0_dot_metric_matches_brute_force(db, elem):
    rng = random.Random(1)
    D = 24
    rows = [[rng.randint(-20, 20) for _ in range(D)] for _ in range(200)]
    conv = "vec_int8(?)" if elem == "int8" else "?"
    db.execute(f"create virtual table t using vec0(v {elem}[{D}] distance_metric=dot)")
    for i, r in enumerate(rows):
        db.execute(f"insert into t(rowid, v) values (?, {conv})", [i, str(r)])
    q = [rng.randint(-20, 20) for _ in range(D)]
    got = db.execute(
        f"select rowid, distance from t where v match {conv} and k = 5", [str(q)]
    ).fetchall()
    want = sorted(((i, -sum(a * b for a, b in zip(r, q))) for i, r in enumerate(rows)), key=lambda x: x[1])[:5]
    assert [d for _, d in got] == [float(d) for _, d in want]
