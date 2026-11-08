"""Compare native capture mechanisms on new disposable fixtures, not project data."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import statistics
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(command, expected=0):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True, timeout=120)
    if result.returncode != expected:
        raise RuntimeError(f'{command}: exit {result.returncode}\n{result.stdout}\n{result.stderr}')
    return result.stdout


def oracle(path, rows, updates, mode, committed=True):
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA integrity_check').fetchone() == ('ok',)
        rev = 1 + updates if committed else 1
        body = ('final' if updates == 2 else 'middle') if committed else 'initial'
        expected = [(key, rev, body, b'\x00\xff') for key in range(1, rows + 1)]
        assert db.execute('SELECT * FROM tasks ORDER BY id').fetchall() == expected
        batches = (rows + 99) // 100 if committed else 0
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == batches
        assert db.execute('SELECT count(*) FROM context').fetchone()[0] == 0
        assert db.execute('SELECT count(*) FROM captured').fetchone()[0] == (batches if mode == 'session' else 0)
        assert db.execute('SELECT count(*) FROM audit').fetchone()[0] == (rows if mode == 'log' and committed else 0)
        if mode == 'log' and committed:
            expected = [(key,1,rev,'initial',body,b'\x00\xff',b'\x00\xff') for key in range(1,rows+1)]
            assert db.execute('SELECT id,old_rev,new_rev,old_body,new_body,old_blob,new_blob FROM audit ORDER BY id').fetchall() == expected


def main():
    if not __debug__:
        raise RuntimeError('this experiment requires Python assertions')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='vec-history-') as temporary:
        work = Path(temporary)
        binary = work / 'history'
        flags = ['-O2','-Wall','-Wextra','-DSQLITE_ENABLE_SESSION','-DSQLITE_ENABLE_PREUPDATE_HOOK','-DSQLITE_ENABLE_FTS5']
        command = ['cc',*flags,'-I'+str(REPO/'vendor'),HERE/'history.c',REPO/'vendor/sqlite3.c','-lm','-o',binary]
        if sys.platform.startswith('linux'):
            command += ['-ldl','-lpthread']
        call(command)
        checks = json.loads(call([binary,'verify']))
        recovery = []
        for mode in ['session','log']:
            for committed in [False,True]:
                path = work / f'recovery-{mode}-{committed}.db'
                action = 'commit_' if committed else 'crash_'
                call([binary,action+mode,path,100,1],expected=0 if committed else 17)
                oracle(path,100,1,mode,committed)
                recovery.append(dict(mode=mode,committed=committed,validated=True))
        samples = []
        modes = ['off','session','log']
        for repeat in range(3):
            for updates in [1,2]:
                for mode in modes[repeat:]+modes[:repeat]:
                    path = work / f'bench-{repeat}-{updates}-{mode}.db'
                    row = json.loads(call([binary,mode,path,5000,updates]))
                    oracle(path,5000,updates,mode)
                    row['repeat'] = repeat
                    samples.append(row)
        summary = []
        for updates in [1,2]:
            for mode in modes:
                rows = [r for r in samples if r['mode']==mode and r['updates']==updates]
                summary.append(dict(mode=mode,updates=updates,
                    median_ms=statistics.median(r['seconds'] for r in rows)*1000,
                    database_bytes=rows[0]['database_bytes'],changeset_bytes=rows[0]['changeset_bytes'],
                    sqlite_peak_bytes=max(r['sqlite_peak_bytes'] for r in rows),
                    session_peak_bytes=max(r['session_peak_bytes'] for r in rows)))
        result = dict(base_revision=call(['git','-C',REPO,'rev-parse','HEAD']).strip(),
            platform=platform.platform(),oracle_sqlite=sqlite3.sqlite_version,
            compiler=call(['cc','--version']).splitlines()[0],flags=flags,
            inputs={str(p.relative_to(REPO)):sha(p) for p in [HERE/'history.c',HERE/'run.py',REPO/'vendor/sqlite3.c',REPO/'vendor/sqlite3.h']},
            checks=checks,recovery=recovery,samples=samples,summary=summary)
        (args.output/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
