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


@pytest.mark.parametrize("scale", [1e-25, 1e-20, 1.0, 1e18, 3e37])
def test_cosine_high_dim_extreme_magnitudes(db, scale):
    """>16 dims takes the SIMD path; tiny/huge vectors must fall back exactly."""
    D = 64
    a = [scale * (1 + (i % 3)) for i in range(D)]
    b = [x * 2 for x in a]          # parallel: distance 0
    c = [-x for x in a]             # opposite: distance 2
    d = db.execute("select vec_distance_cosine(?, ?), vec_distance_cosine(?, ?)",
                   [_f32(a), _f32(b), _f32(a), _f32(c)]).fetchone()
    assert d[0] == pytest.approx(0.0, abs=1e-6)
    assert d[1] == pytest.approx(2.0, abs=1e-6)
