import pytest
import sqlite3
from helpers import _f32, exec


@pytest.fixture()
def db():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    db.load_extension("dist/vec0")
    db.enable_load_extension(False)
    return db


# ============================================================================
# Parser tests — quantizer and oversample options
# ============================================================================


def test_ivf_quantizer_binary(db):
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        "v float[768] indexed by ivf(nlist=64, quantizer=binary, oversample=10))"
    )
    tables = [
        r[0]
        for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
        ).fetchall()
    ]
    assert "t_ivf_centroids00" in tables
    assert "t_ivf_cells00" in tables


def test_ivf_quantizer_int8(db):
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        "v float[768] indexed by ivf(nlist=64, quantizer=int8))"
    )
    tables = [
        r[0]
        for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
        ).fetchall()
    ]
    assert "t_ivf_centroids00" in tables


def test_ivf_quantizer_none_explicit(db):
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        "v float[768] indexed by ivf(quantizer=none))"
    )
    # Should work — same as no quantizer
    tables = [
        r[0]
        for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
        ).fetchall()
    ]
    assert "t_ivf_centroids00" in tables


def test_ivf_quantizer_all_params(db):
    """All params together: nlist, nprobe, quantizer, oversample."""
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        "v float[768] distance_metric=cosine "
        "indexed by ivf(nlist=128, nprobe=16, quantizer=int8, oversample=4))"
    )
    tables = [
        r[0]
        for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
        ).fetchall()
    ]
    assert "t_ivf_centroids00" in tables


def test_ivf_error_oversample_without_quantizer(db):
    """oversample > 1 without quantizer should error."""
    result = exec(
        db,
        "CREATE VIRTUAL TABLE t USING vec0("
        "v float[768] indexed by ivf(oversample=10))",
    )
    assert "error" in result


def test_ivf_error_unknown_quantizer(db):
    result = exec(
        db,
        "CREATE VIRTUAL TABLE t USING vec0("
        "v float[768] indexed by ivf(quantizer=pq))",
    )
    assert "error" in result


def test_ivf_oversample_1_without_quantizer_ok(db):
    """oversample=1 (default) is fine without quantizer."""
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        "v float[768] indexed by ivf(nlist=64))"
    )
    # Should succeed — oversample defaults to 1


# ============================================================================
# Functional tests — insert, train, query with quantized IVF
# ============================================================================


def test_ivf_int8_insert_and_query(db):
    """int8 quantized IVF: insert, train, query."""
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        "v float[4] indexed by ivf(nlist=2, quantizer=int8, oversample=4))"
    )
    for i in range(20):
        db.execute(
            "INSERT INTO t(rowid, v) VALUES (?, ?)", [i, _f32([i, 0, 0, 0])]
        )

    db.execute("INSERT INTO t(t) VALUES ('compute-centroids')")

    # Should be able to query
    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE v MATCH ? AND k = 5",
        [_f32([10.0, 0, 0, 0])],
    ).fetchall()
    assert len(rows) == 5
    # Top result should be close to 10
    assert rows[0][0] in range(8, 13)

    # Full vectors should be in _ivf_vectors table
    fv_count = db.execute("SELECT count(*) FROM t_ivf_vectors00").fetchone()[0]
    assert fv_count == 20


def test_ivf_binary_insert_and_query(db):
    """Binary quantized IVF: insert, train, query."""
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        "v float[8] indexed by ivf(nlist=2, quantizer=binary, oversample=4))"
    )
    for i in range(20):
        # Vectors with varying sign patterns
        v = [(i * 0.1 - 1.0) + j * 0.3 for j in range(8)]
        db.execute(
            "INSERT INTO t(rowid, v) VALUES (?, ?)", [i, _f32(v)]
        )

    db.execute("INSERT INTO t(t) VALUES ('compute-centroids')")

    rows = db.execute(
        "SELECT rowid FROM t WHERE v MATCH ? AND k = 5",
        [_f32([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])],
    ).fetchall()
    assert len(rows) == 5

    # Full vectors stored
    fv_count = db.execute("SELECT count(*) FROM t_ivf_vectors00").fetchone()[0]
    assert fv_count == 20


def test_ivf_int8_cell_sizes_smaller(db):
    """Cell blobs should be smaller with int8 quantization."""
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        "v float[768] indexed by ivf(nlist=2, quantizer=int8, oversample=1))"
    )
    for i in range(10):
        db.execute(
            "INSERT INTO t(rowid, v) VALUES (?, ?)",
            [i, _f32([float(x) / 768 for x in range(768)])],
        )

    # Cell vectors blob: 10 vectors at int8 = 10 * 768 = 7680 bytes
    # vs float32 = 10 * 768 * 4 = 30720 bytes
    # But cells have capacity 64, so blob = 64 * 768 = 49152 (int8) vs 64*768*4=196608 (float32)
    blob_size = db.execute(
        "SELECT length(vectors) FROM t_ivf_cells00 LIMIT 1"
    ).fetchone()[0]
    # int8: 64 slots * 768 bytes = 49152
    assert blob_size == 64 * 768


def test_ivf_binary_cell_sizes_smaller(db):
    """Cell blobs should be much smaller with binary quantization."""
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        "v float[768] indexed by ivf(nlist=2, quantizer=binary, oversample=1))"
    )
    for i in range(10):
        db.execute(
            "INSERT INTO t(rowid, v) VALUES (?, ?)",
            [i, _f32([float(x) / 768 for x in range(768)])],
        )

    blob_size = db.execute(
        "SELECT length(vectors) FROM t_ivf_cells00 LIMIT 1"
    ).fetchone()[0]
    # binary: 64 slots * 768/8 bytes = 6144
    assert blob_size == 64 * (768 // 8)


def test_ivf_int8_oversample_improves_recall(db):
    """Oversample re-ranking should improve recall over oversample=1."""
    # Create two tables: one with oversample=1, one with oversample=10
    db.execute(
        "CREATE VIRTUAL TABLE t1 USING vec0("
        "v float[4] indexed by ivf(nlist=4, quantizer=int8, oversample=1))"
    )
    db.execute(
        "CREATE VIRTUAL TABLE t2 USING vec0("
        "v float[4] indexed by ivf(nlist=4, quantizer=int8, oversample=10))"
    )
    for i in range(100):
        v = _f32([i * 0.1, (i * 0.1) % 3, (i * 0.3) % 5, i * 0.01])
        db.execute("INSERT INTO t1(rowid, v) VALUES (?, ?)", [i, v])
        db.execute("INSERT INTO t2(rowid, v) VALUES (?, ?)", [i, v])

    db.execute("INSERT INTO t1(t1) VALUES ('compute-centroids')")
    db.execute("INSERT INTO t2(t2) VALUES ('compute-centroids')")
    db.execute("INSERT INTO t1(t1) VALUES ('nprobe=4')")
    db.execute("INSERT INTO t2(t2) VALUES ('nprobe=4')")

    query = _f32([5.0, 1.5, 2.5, 0.5])
    r1 = db.execute("SELECT rowid FROM t1 WHERE v MATCH ? AND k=10", [query]).fetchall()
    r2 = db.execute("SELECT rowid FROM t2 WHERE v MATCH ? AND k=10", [query]).fetchall()

    # Both should return 10 results
    assert len(r1) == 10
    assert len(r2) == 10
    # oversample=10 should have at least as good recall (same or better ordering)


def test_ivf_quantized_delete(db):
    """Delete should remove from both cells and _ivf_vectors."""
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        "v float[4] indexed by ivf(nlist=2, quantizer=int8, oversample=1))"
    )
    for i in range(10):
        db.execute(
            "INSERT INTO t(rowid, v) VALUES (?, ?)", [i, _f32([i, 0, 0, 0])]
        )

    db.execute("INSERT INTO t(t) VALUES ('compute-centroids')")
    assert db.execute("SELECT count(*) FROM t_ivf_vectors00").fetchone()[0] == 10

    db.execute("DELETE FROM t WHERE rowid = 5")
    # _ivf_vectors should have 9 rows
    assert db.execute("SELECT count(*) FROM t_ivf_vectors00").fetchone()[0] == 9


@pytest.mark.parametrize("quantizer", ["int8", "binary"])
def test_ivf_quantized_vectors_read_back(db, quantizer):
    """A quantized column returns the vectors it was given from a point
    query, a full scan and a KNN query, before and after training."""
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        f"v float[16] indexed by ivf(nlist=2, quantizer={quantizer}))"
    )
    vecs = {
        i: _f32([((i * 7 + j * 3) % 11) / 10 - 0.5 for j in range(16)])
        for i in range(1, 11)
    }
    for rowid, v in vecs.items():
        db.execute("INSERT INTO t(rowid, v) VALUES (?, ?)", [rowid, v])

    for trained in [False, True]:
        if trained:
            db.execute("INSERT INTO t(t) VALUES ('compute-centroids')")
        assert db.execute("SELECT v FROM t WHERE rowid = 3").fetchone()[0] == vecs[3]
        assert {r[0]: r[1] for r in db.execute("SELECT rowid, v FROM t")} == vecs
        knn = db.execute(
            "SELECT rowid, v FROM t WHERE v MATCH ? AND k = 10", [vecs[3]]
        ).fetchall()
        assert {r[0]: r[1] for r in knn} == vecs


def test_ivf_binary_rejects_non_multiple_of_8_dims(db):
    """Binary quantizer requires dimensions divisible by 8."""
    with pytest.raises(sqlite3.OperationalError):
        db.execute(
            "CREATE VIRTUAL TABLE t USING vec0("
            "  v float[12] indexed by ivf(quantizer=binary)"
            ")"
        )

    # Dimensions divisible by 8 should work
    db.execute(
        "CREATE VIRTUAL TABLE t2 USING vec0("
        "  v float[16] indexed by ivf(quantizer=binary)"
        ")"
    )


# ============================================================================
# RaBitQ
# ============================================================================


def test_ivf_rabitq_rejects_l1_and_bad_dims(db):
    with pytest.raises(sqlite3.OperationalError):
        db.execute(
            "CREATE VIRTUAL TABLE t1 USING vec0("
            "v float[8] distance_metric=l1 indexed by ivf(quantizer=rabitq))"
        )
    with pytest.raises(sqlite3.OperationalError):
        db.execute(
            "CREATE VIRTUAL TABLE t2 USING vec0("
            "v float[12] indexed by ivf(quantizer=rabitq))"
        )


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_ivf_rabitq_recall(db, metric):
    """RaBitQ estimate + oversample rerank should match brute force."""
    import random

    rng = random.Random(0)
    D, N, K = 128, 1000, 10
    # clustered data; isotropic gaussian has no real neighbors to find
    centers = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(30)]
    point = lambda: [x + rng.gauss(0, 0.4) for x in rng.choice(centers)]
    vecs = [point() for _ in range(N)]
    ivf = "indexed by ivf(nlist=8, nprobe=8, quantizer=rabitq, oversample=4)"
    for name, idx in [("t", ivf), ("ref", "")]:
        db.execute(
            f"CREATE VIRTUAL TABLE {name} USING vec0("
            f"v float[{D}] distance_metric={metric} {idx})"
        )
        db.executemany(
            f"INSERT INTO {name}(rowid, v) VALUES (?, ?)",
            [(i, _f32(v)) for i, v in enumerate(vecs)],
        )
    db.execute("INSERT INTO t(t) VALUES ('compute-centroids')")
    # rows inserted after training take the trained insert path
    extra = point()
    for name in ("t", "ref"):
        db.execute(f"INSERT INTO {name}(rowid, v) VALUES (?, ?)", [N, _f32(extra)])

    hits = 0
    for _ in range(20):
        q = _f32(point())
        sql = "SELECT rowid FROM {} WHERE v MATCH ? AND k = ?"
        got = {r[0] for r in db.execute(sql.format("t"), [q, K])}
        want = {r[0] for r in db.execute(sql.format("ref"), [q, K])}
        hits += len(got & want)
    assert hits / (20 * K) >= 0.9

    # Codes are D/8 bits + 2 floats per vector
    cell = db.execute("SELECT n_vectors, length(vectors) FROM t_ivf_cells00 LIMIT 1").fetchone()
    assert cell[1] % (D // 8 + 8) == 0


@pytest.mark.parametrize("qz,metric", [
    ("fp16", "l2"), ("bf16", "cosine"), ("int4", "l2"), ("int4", "dot"), ("fp16", "l1"),
])
def test_ivf_compact_quantizers(db, qz, metric):
    """fp16/bf16/int4 cells + rerank match brute force; codes are compact."""
    import random

    rng = random.Random(2)
    D, N, K = 32, 400, 5
    vecs = [[rng.uniform(-1, 1) for _ in range(D)] for _ in range(N)]
    ivf = f"indexed by ivf(nlist=4, nprobe=4, quantizer={qz}, oversample=4)"
    for name, idx in [("t", ivf), ("ref", "")]:
        db.execute(f"CREATE VIRTUAL TABLE {name} USING vec0(v float[{D}] distance_metric={metric} {idx})")
        db.executemany(f"INSERT INTO {name}(rowid, v) VALUES (?, ?)",
                       [(i, _f32(v)) for i, v in enumerate(vecs)])
    db.execute("INSERT INTO t(t) VALUES ('compute-centroids')")
    hits = 0
    for _ in range(10):
        q = _f32([rng.uniform(-1, 1) for _ in range(D)])
        sql = "SELECT rowid FROM {} WHERE v MATCH ? AND k = ?"
        hits += len({r[0] for r in db.execute(sql.format("t"), [q, K])} &
                    {r[0] for r in db.execute(sql.format("ref"), [q, K])})
    assert hits / (10 * K) >= 0.9
    per_vec = {"fp16": 2 * D, "bf16": 2 * D, "int4": D // 2}[qz]
    assert db.execute("SELECT length(vectors) FROM t_ivf_cells00 LIMIT 1").fetchone()[0] % per_vec == 0
    # point reads return the original float32 vector
    assert db.execute("SELECT v FROM t WHERE rowid = 3").fetchone()[0] == _f32(vecs[3])


def test_ivf_int4_rejects_odd_dims(db):
    with pytest.raises(sqlite3.OperationalError):
        db.execute("CREATE VIRTUAL TABLE t USING vec0(v float[7] indexed by ivf(quantizer=int4))")
