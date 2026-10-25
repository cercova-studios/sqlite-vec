import sqlite3
import struct

import pytest


def _f32(values):
    return struct.pack(f"{len(values)}f", *values)


SUPPORTS_VTAB_LIMIT = sqlite3.sqlite_version_info[1] >= 41

pytestmark = pytest.mark.skipif(
    not SUPPORTS_VTAB_LIMIT, reason="SQLITE_INDEX_CONSTRAINT_LIMIT needs SQLite >= 3.41"
)


@pytest.fixture()
def v(db):
    db.execute("create virtual table v using vec0(e float[2])")
    for i in range(1, 11):
        db.execute("insert into v(rowid, e) values (?, ?)", [i, _f32([float(i), 0.0])])
    return db


Q = _f32([0.0, 0.0])


def rowids(db, sql, args=()):
    return [r[0] for r in db.execute(sql, args)]


def test_order_by_distance_limit_matches_k(v):
    assert rowids(
        v, "select rowid from v where e match ? order by distance limit 3", [Q]
    ) == rowids(v, "select rowid from v where e match ? and k = 3", [Q]) == [1, 2, 3]


def test_limit_without_order_by_is_still_bounded(v):
    # SQLite pushes the LIMIT constraint even without ORDER BY; the KNN
    # scan returns nearest-first, so the rows are the same.
    assert rowids(v, "select rowid from v where e match ? limit 3", [Q]) == [1, 2, 3]


def test_limit_offset_returns_the_window(v):
    assert rowids(
        v, "select rowid from v where e match ? order by distance limit 3 offset 2", [Q]
    ) == [3, 4, 5]
    assert rowids(
        v, "select rowid from v where e match ? order by distance limit 2, 3", [Q]
    ) == [3, 4, 5]


def test_limit_offset_parameters(v):
    assert rowids(
        v,
        "select rowid from v where e match ? order by distance limit ? offset ?",
        [Q, 3, 2],
    ) == [3, 4, 5]


def test_limit_offset_beyond_end_is_empty(v):
    assert (
        rowids(
            v, "select rowid from v where e match ? order by distance limit 3 offset 20", [Q]
        )
        == []
    )


def test_limit_zero_and_negative_offset(v):
    assert rowids(v, "select rowid from v where e match ? order by distance limit 0", [Q]) == []
    # SQLite treats a negative OFFSET as 0.
    assert rowids(
        v, "select rowid from v where e match ? order by distance limit 3 offset -1", [Q]
    ) == [1, 2, 3]


def test_both_limit_and_k_is_an_error(v):
    with pytest.raises(sqlite3.OperationalError, match="not both"):
        v.execute("select rowid from v where e match ? and k = 3 order by distance limit 5", [Q])


def test_neither_limit_nor_k_is_an_error(v):
    with pytest.raises(sqlite3.OperationalError, match="is required on vec0 knn queries"):
        v.execute("select rowid from v where e match ?", [Q])


def test_negative_limit_is_an_error(v):
    # LIMIT -1 means "no limit" in SQLite; a KNN query must stay bounded.
    with pytest.raises(sqlite3.OperationalError, match="greater than or equal to 0"):
        v.execute("select rowid from v where e match ? order by distance limit -1", [Q])


def test_limit_plus_offset_over_cap_is_an_error(v):
    with pytest.raises(sqlite3.OperationalError, match="too large"):
        v.execute(
            "select rowid from v where e match ? order by distance limit 4000 offset 200", [Q]
        )


def test_limit_with_rowid_in_and_distance_constraint(v):
    assert rowids(
        v,
        "select rowid from v where e match ? and rowid in (2,3,4,5,6) order by distance limit 2 offset 1",
        [Q],
    ) == [3, 4]
    assert rowids(
        v,
        "select rowid from v where e match ? and distance < 5 order by distance limit 10",
        [Q],
    ) == [1, 2, 3, 4]
