import sqlite3
import struct

import pytest


def _write(path, fmt, vectors):
    with open(path, "wb") as f:
        for v in vectors:
            f.write(struct.pack("<i", len(v)))
            f.write(struct.pack(f"<{len(v)}{fmt}", *v))


def test_fvecs_roundtrip(db, tmp_path):
    path = str(tmp_path / "data.fvecs")
    _write(path, "f", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rows = db.execute(
        "select rowid, vec_to_json(vector) from vec_file_each(?)", [path]
    ).fetchall()
    assert [(r[0], r[1]) for r in rows] == [
        (0, "[1.000000,2.000000,3.000000]"),
        (1, "[4.000000,5.000000,6.000000]"),
    ]


def test_fvecs_loads_into_vec0_and_searches(db, tmp_path):
    path = str(tmp_path / "data.fvecs")
    _write(path, "f", [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    db.execute("create virtual table v using vec0(e float[2])")
    db.execute("insert into v(rowid, e) select rowid + 1, vector from vec_file_each(?)", [path])
    got = [
        r[0]
        for r in db.execute(
            "select rowid from v where e match ? order by distance limit 2",
            [struct.pack("<2f", 1.0, 0.0)],
        )
    ]
    assert got == [1, 3]


def test_bvecs_and_ivecs(db, tmp_path):
    bpath = str(tmp_path / "data.bvecs")
    _write(bpath, "B", [[1, 2, 255], [0, 0, 0]])
    rows = db.execute("select rowid, vector from vec_file_each(?)", [bpath]).fetchall()
    assert [(r[0], bytes(r[1])) for r in rows] == [
        (0, bytes([1, 2, 255])),
        (1, bytes([0, 0, 0])),
    ]

    ipath = str(tmp_path / "data.ivecs")
    _write(ipath, "i", [[7, -8], [9, 10]])
    rows = db.execute("select rowid, vector from vec_file_each(?)", [ipath]).fetchall()
    assert [(r[0], struct.unpack("<2i", bytes(r[1]))) for r in rows] == [
        (0, (7, -8)),
        (1, (9, 10)),
    ]


def test_empty_file_is_zero_rows(db, tmp_path):
    path = str(tmp_path / "empty.fvecs")
    open(path, "wb").close()
    assert db.execute("select count(*) from vec_file_each(?)", [path]).fetchone()[0] == 0


def test_missing_file_is_an_error(db, tmp_path):
    with pytest.raises(sqlite3.OperationalError, match="could not open vector file"):
        db.execute(
            "select * from vec_file_each(?)", [str(tmp_path / "nope.fvecs")]
        ).fetchall()


def test_unknown_extension_is_an_error(db, tmp_path):
    path = str(tmp_path / "data.npy")
    _write(path, "f", [[1.0]])
    with pytest.raises(sqlite3.OperationalError, match="unsupported vector file"):
        db.execute("select * from vec_file_each(?)", [path]).fetchall()


def test_truncated_file_is_an_error(db, tmp_path):
    path = str(tmp_path / "bad.fvecs")
    with open(path, "wb") as f:
        f.write(struct.pack("<i", 3))
        f.write(struct.pack("<2f", 1.0, 2.0))
    with pytest.raises(sqlite3.OperationalError, match="truncated"):
        db.execute("select * from vec_file_each(?)", [path]).fetchall()


def test_inconsistent_dimension_is_an_error(db, tmp_path):
    path = str(tmp_path / "bad.fvecs")
    with open(path, "wb") as f:
        # Total size stays a multiple of the first (12-byte) record size, so
        # the dimension mismatch is what surfaces: 12 + 16 + 20 = 48 bytes.
        f.write(struct.pack("<i", 2))
        f.write(struct.pack("<2f", 1.0, 2.0))
        f.write(struct.pack("<i", 3))
        f.write(struct.pack("<3f", 1.0, 2.0, 3.0))
        f.write(struct.pack("<i", 4))
        f.write(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
    with pytest.raises(sqlite3.OperationalError, match="has dimension"):
        db.execute("select * from vec_file_each(?)", [path]).fetchall()
