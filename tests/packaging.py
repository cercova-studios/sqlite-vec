"""Build isolated consumers; never overwrite a developer's generated artifacts."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[1]
BASELINE = False
PROBE = r'''
#include <assert.h>
#include <string.h>
#include "sqlite3.h"
#include "sqlite-vec.h"
int main(void) {
  sqlite3 *db = 0;
  sqlite3_stmt *stmt = 0;
  assert(sqlite3_open(":memory:", &db) == SQLITE_OK);
  assert(sqlite3_vec_init(db, 0, 0) == SQLITE_OK);
  assert(sqlite3_exec(db,
      "CREATE VIRTUAL TABLE t USING vec0(v float[2] indexed by hnsw(m=4));"
      "INSERT INTO t(rowid,v) VALUES(1,'[0,0]'),(2,'[1,0]');"
      "CREATE TABLE e(id,src,dst); INSERT INTO e VALUES(7,1,2);",
      0, 0, 0) == SQLITE_OK);
  assert(sqlite3_prepare_v2(db,
      "SELECT rowid FROM t WHERE v MATCH '[1,0]' AND k=1", -1, &stmt, 0) == SQLITE_OK);
  assert(sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_int64(stmt, 0) == 2);
  sqlite3_finalize(stmt);
  assert(sqlite3_prepare_v2(db,
      "SELECT graph_witness('e','src','dst','id',1,2,1000,65536)", -1, &stmt, 0) == SQLITE_OK);
  assert(sqlite3_step(stmt) == SQLITE_ROW);
  assert(strcmp((const char *)sqlite3_column_text(stmt, 0), "[[1,null],[2,7]]") == 0);
  sqlite3_finalize(stmt);
  assert(sqlite3_close(db) == SQLITE_OK);
}
'''


class Packaging(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='sqlite-vec-package-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        names = subprocess.check_output(['git','ls-files','sqlite-vec*.c'],cwd=REPO,text=True).splitlines()
        names.append('examples/sqlite3-cli/core_init.c')
        # Committed native inputs isolate build-recipe regressions from C edits.
        for name in names:
            data = subprocess.check_output(['git','show','HEAD:'+name],cwd=REPO)
            (self.root/name).parent.mkdir(parents=True,exist_ok=True)
            (self.root/name).write_bytes(data)
        for name in ['Makefile','VERSION','sqlite-vec.h.tmpl','scripts/amalgamate.py',
                     'bindings/rust/Makefile','bindings/rust/Cargo.toml.tmpl']:
            path = self.root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(subprocess.check_output(['git','show','HEAD:'+name],cwd=REPO) if BASELINE else (REPO/name).read_bytes())
        (self.root/'vendor').mkdir()
        for name in ['sqlite3.c','sqlite3.h','sqlite3ext.h','shell.c']:
            shutil.copyfile(REPO/'vendor'/name,self.root/'vendor'/name)
        shutil.copyfile(REPO/'sqlite-vec.h',self.root/'sqlite-vec.h')

    def run_command(self, command, cwd=None, expected=0):
        result = subprocess.run(command,cwd=cwd or self.root,capture_output=True,text=True)
        self.assertEqual(result.returncode,expected,result.stdout+'\n'+result.stderr)
        return result

    def consumer(self, folder, source):
        (folder/'consumer.c').write_text(PROBE)
        command = ['cc','-DSQLITE_CORE','-DSQLITE_VEC_STATIC','-I'+str(folder),
                   '-I'+str(self.root/'vendor'),str(folder/'consumer.c'),str(source),
                   str(self.root/'vendor/sqlite3.c'),'-lm','-o',str(folder/'consumer')]
        if sys.platform.startswith('linux'):
            command += ['-ldl','-lpthread']
        self.run_command(command)
        self.run_command([str(folder/'consumer')])

    def test_rust_staging_is_standalone_and_tracks_features(self):
        rust = self.root/'bindings/rust'
        self.run_command(['make','deps'],cwd=rust)
        self.consumer(rust,rust/'sqlite-vec.c')
        self.run_command(['make','-q','deps'],cwd=rust)
        feature = self.root/'sqlite-vec-graph.c'
        now = time.time_ns()+2_000_000_000
        os.utime(feature,ns=(now,now))
        self.run_command(['make','-q','deps'],cwd=rust,expected=1)

    def test_static_consumer_and_incremental_inputs(self):
        self.run_command(['make','static'])
        dist = self.root/'dist'
        # Normalize directories to isolate file dependencies from incidental mtimes.
        past = time.time()-100
        for folder in [dist,dist/'.objs']:
            os.utime(folder,(past,past))
        self.run_command(['make','-q','static'])
        self.assertTrue((dist/'sqlite-vec.h').is_file())
        self.consumer(dist,dist/'libsqlite_vec0.a')

    def test_native_builds_track_included_modules(self):
        self.run_command(['make','static','cli'])
        dist = self.root/'dist'
        past = time.time()-100
        for folder in [dist,dist/'.objs']:
            os.utime(folder,(past,past))
        self.run_command(['make','-q','static','dist/sqlite3'])
        os.utime(dist,(past,past))
        feature = self.root/'sqlite-vec-graph.c'
        now = time.time_ns()+2_000_000_000
        os.utime(feature,ns=(now,now))
        self.run_command(['make','-q','static'],expected=1)
        self.run_command(['make','-q','dist/sqlite3'],expected=1)

    def test_cli_builds_once_and_skips_unchanged_inputs(self):
        compiler = shutil.which('cc')
        self.assertIsNotNone(compiler)
        calls = self.root/'compiler-calls.jsonl'
        wrapper = self.root/'record-cc'
        wrapper.write_text(
            '#!/usr/bin/env python3\nimport json, os, sys\n'
            f'with open({str(calls)!r}, "a") as log:\n'
            '    log.write(json.dumps(sys.argv[1:]) + "\\n")\n'
            f'os.execv({compiler!r}, [{compiler!r}, *sys.argv[1:]])\n')
        wrapper.chmod(0o755)
        command = ['make',f'CC={wrapper}','cli']
        self.run_command(command)
        recorded = [json.loads(line) for line in calls.read_text().splitlines()]
        builds = [args for args in recorded if '-o' in args]
        links = [args for args in builds if args[args.index('-o')+1] == 'dist/sqlite3']
        self.assertEqual(len(links),1,'CLI output must be compiled/linked exactly once')
        self.assertEqual(len(builds),1,'CLI must not build unused intermediate objects')
        target = self.root/'dist/sqlite3'
        identity = (target.stat().st_mtime_ns,target.read_bytes())
        self.run_command(command)
        repeated = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertEqual([args for args in repeated if '-o' in args],builds)
        self.assertEqual((target.stat().st_mtime_ns,target.read_bytes()),identity)
        os.utime(target.parent,None)
        self.run_command(['make','-q','dist/sqlite3'])
        for name in ['sqlite-vec.h','vendor/sqlite3.c','vendor/sqlite3.h',
                     'vendor/sqlite3ext.h','vendor/shell.c','examples/sqlite3-cli/core_init.c']:
            source = self.root/name
            previous = source.stat()
            changed = time.time_ns()+2_000_000_000
            try:
                os.utime(source,ns=(changed,changed))
                self.run_command(['make','-q','dist/sqlite3'],expected=1)
            finally:
                os.utime(source,ns=(previous.st_atime_ns,previous.st_mtime_ns))
            self.run_command(['make','-q','dist/sqlite3'])
        result = self.run_command([str(target),'-batch','-bail',':memory:',
                                   'SELECT sqlite_version(),vec_version();'])
        self.assertEqual(result.stdout.strip(),
                         '3.45.3|v'+(self.root/'VERSION').read_text().strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline',action='store_true',help='run the same checks against committed build recipes')
    args,remaining = parser.parse_known_args()
    BASELINE = args.baseline
    unittest.main(argv=[sys.argv[0],*remaining])
