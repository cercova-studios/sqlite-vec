#!/usr/bin/env python3
"""Disposable local task search using this project's installed fork bundle."""
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent
SQLITE = ROOT / '.sqlite-vec/bin/sqlite3'


def sql(database, statements):
    result = subprocess.run(
        [str(SQLITE), '-json', str(database)],
        input='.bail on\n' + statements + '\n', text=True,
        capture_output=True, check=True, cwd=database.parent,
    )
    return json.loads(result.stdout) if result.stdout.strip() else []


def check(database):
    canonical = sql(database, 'SELECT id,title,embedding FROM tasks ORDER BY id;')
    assert len(canonical) == 4, canonical
    expected = sorted(
        [(row['id'], row['title'], math.dist(json.loads(row['embedding']), [0, 0]))
         for row in canonical], key=lambda row: (row[2], row[0]),
    )
    assert [row[0] for row in expected] == [101, 202, 303, 404]
    assert [row[2] for row in expected] == [0.0, 5.0, 10.0, 13.0]
    actual = sql(database, """
        SELECT tasks.id,tasks.title,vectors.distance
        FROM vectors JOIN tasks ON tasks.id=vectors.rowid
        WHERE vectors.embedding MATCH '[0,0]' AND k=4
        ORDER BY vectors.distance,tasks.id;
    """)
    assert len(actual) == len(expected), actual
    for found, wanted in zip(actual, expected):
        assert (found['id'], found['title']) == wanted[:2], (found, wanted)
        assert math.isclose(found['distance'], wanted[2], abs_tol=1e-6), (found, wanted)
    assert sql(database, 'PRAGMA integrity_check;') == [{'integrity_check': 'ok'}]
    return actual


def main():
    started = time.monotonic()
    data_root = ROOT / 'runs'
    data_root.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='demo-', dir=data_root))
    database = run / 'app.db'
    sql(database, """
        CREATE TABLE tasks(id INTEGER PRIMARY KEY,title TEXT NOT NULL,embedding TEXT NOT NULL);
        CREATE VIRTUAL TABLE vectors USING vec0(embedding float[2]);
        BEGIN;
        INSERT INTO tasks VALUES
          (101,'Origin','[0,0]'),(202,'Nearby','[3,4]'),
          (303,'Farther','[6,8]'),(404,'Farthest','[5,12]');
        INSERT INTO vectors(rowid,embedding) SELECT id,embedding FROM tasks;
        COMMIT;
    """)
    initial = check(database)
    visible = sql(database, """
        BEGIN;
        INSERT INTO tasks VALUES(999,'Transient','[-1,0]');
        INSERT INTO vectors(rowid,embedding) VALUES(999,'[-1,0]');
        SELECT rowid,distance FROM vectors WHERE embedding MATCH '[-1,0]' AND k=1;
        ROLLBACK;
    """)
    assert visible == [{'rowid': 999, 'distance': 0.0}], visible
    assert check(database) == initial
    sql(database, """
        BEGIN;
        SAVEPOINT temporary;
        INSERT INTO tasks VALUES(888,'Savepoint','[-2,0]');
        INSERT INTO vectors(rowid,embedding) VALUES(888,'[-2,0]');
        ROLLBACK TO temporary;
        RELEASE temporary;
        COMMIT;
    """)
    assert check(database) == initial
    # Each sql() invocation opens a new process and therefore a new connection.
    assert sql(database, 'SELECT count(*) AS n FROM tasks WHERE id IN (888,999);') == [{'n': 0}]
    restored = run / 'restored.db'
    sql(database, '.backup backup.db')
    sql(restored, '.restore backup.db')
    assert check(restored) == initial
    print(json.dumps({'ok': True, 'database': str(database), 'restored': str(restored),
                      'neighbors': initial, 'reopen': True, 'rollback': True,
                      'savepoint': True, 'backup_restore': True,
                      'elapsed_seconds': round(time.monotonic() - started, 3)}))


if __name__ == '__main__':
    if not __debug__:
        raise SystemExit('Run without -O so the example checks remain enabled.')
    main()
