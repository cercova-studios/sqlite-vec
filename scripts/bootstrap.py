#!/usr/bin/env python3
"""Inspect prerequisites or build and check a contributor checkout (macOS/Linux)."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


class Parser(argparse.ArgumentParser):
    def error(self, message):
        print(json.dumps(dict(ok=False, error='usage', message=message,
                              recovery='Run python3 scripts/bootstrap.py --help.')))
        raise SystemExit(2)


def inspect():
    tools = {name: shutil.which(name) for name in ('git', 'make', 'cc')}
    if not (REPO / 'sqlite-vec.h').is_file():
        tools['envsubst'] = shutil.which('envsubst')
    missing = [name for name, path in tools.items() if path is None]
    missing += [str(path) for path in (
        Path('vendor') / name for name in ('sqlite3.c', 'sqlite3.h', 'sqlite3ext.h', 'shell.c')
    ) if not (REPO / path).is_file()]
    if sys.platform not in ('darwin', 'linux'):
        missing.append('macOS or Linux host')
    return dict(repo=str(REPO), executable=str(Path(__file__).resolve()),
                tools=tools, missing=missing,
                guide=str(REPO / 'docs/agent-onboarding.md'))


def develop(output):
    # Make treats whitespace and metacharacters in target names as syntax.
    if any(not (c.isascii() and (c.isalnum() or c in '/._-')) for c in str(output)):
        raise ValueError('output path may contain only ASCII letters, digits, /, ., _, and -')
    output.mkdir(parents=True, exist_ok=False)
    log = output / 'bootstrap.log'
    command = ['make', f'prefix={output}', 'loadable', 'cli', 'test-unit']
    env = os.environ.copy()
    # Environment flags let Make append its host SIMD settings.
    env['CFLAGS'] = '-DSQLITE_ENABLE_FTS5'
    env['GIT_TERMINAL_PROMPT'] = '0'
    # Keep inherited Make goals/flags from changing this bootstrap's targets.
    for key in ('MAKEFLAGS', 'MFLAGS', 'MAKEFILES'):
        env.pop(key, None)
    with log.open('w') as stream:
        stream.write(json.dumps(command) + '\n')
        stream.flush()
        with subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=stream,
                              start_new_session=True) as process:
            try:
                process.wait(timeout=600)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        sql = ("CREATE VIRTUAL TABLE v USING vec0(embedding float[2]);"
               "INSERT INTO v(rowid,embedding) VALUES(1,'[1,0]'),(2,'[0,1]');"
               "SELECT rowid,distance FROM v WHERE embedding MATCH '[1,0]' AND k=1;"
               "CREATE VIRTUAL TABLE f USING fts5(text);"
               "INSERT INTO f VALUES('bootstrap');"
               "SELECT count(*) FROM f WHERE f MATCH 'bootstrap';")
        result = subprocess.run([str(output / 'sqlite3'), '-batch', '-bail', ':memory:', sql],
                                text=True, capture_output=True, timeout=30)
        stream.write(result.stdout + result.stderr)
        result.check_returncode()
        if result.stdout.splitlines() != ['1|0.0', '1']:
            raise ValueError('vector/FTS5 smoke check returned unexpected rows')
    return dict(output=str(output), log=str(log), checks=['test-unit', 'vector-knn', 'fts5'])


def main():
    parser = Parser(description=__doc__)
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('doctor', help='read-only prerequisite report; also the default')
    dev = sub.add_parser('dev', help='build working-tree C sources and run unit/smoke checks')
    dev.add_argument('--output', required=True, type=Path,
                     help='fresh build directory; failure retains bootstrap.log; never overwritten')
    args = parser.parse_args()
    state = inspect()
    if state['missing']:
        print(json.dumps(dict(ok=False, error='prerequisites', **state,
                              recovery='See the guide; install prerequisites explicitly, then retry.')))
        return 1
    if args.command != 'dev':
        print(json.dumps(dict(ok=True, **state,
                              next=[[sys.executable, str(REPO / 'scripts/bootstrap.py'),
                                     'dev', '--output', '/tmp/sqlite-vec-dev'],
                                    [sys.executable, str(REPO / 'scripts/local_bundle.py'), '--help']])))
        return 0
    output = args.output.absolute()
    try:
        result = develop(output)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(ok=False, error='bootstrap_failed', message=str(error),
                              output=str(output), log=str(output / 'bootstrap.log'),
                              recovery='Inspect the log if created; fix the cause and choose a fresh output directory.')))
        return 1
    print(json.dumps(dict(ok=True, repo=str(REPO), **result)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
