"""Manifest failure checks and a relocated installed-bundle journey."""

import argparse
import importlib.util
import json
from pathlib import Path
import select
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('local_bundle', REPO / 'scripts/local_bundle.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)
INSTALLED = None


class Manifest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'payload').write_text('original')
        self.manifest = dict(format='sqlite-vec-local-bundle/1', artifacts={
            'payload': dict(sha256=bundle.digest(self.root / 'payload'), bytes=8)})
        self.save()

    def save(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_hash_size_missing_and_bad_manifest(self):
        self.assertEqual(bundle.verify(self.root), self.manifest)
        (self.root / 'payload').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            bundle.verify(self.root)
        (self.root / 'payload').write_text('longer tampering')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            bundle.verify(self.root)
        (self.root / 'payload').unlink()
        with self.assertRaisesRegex(ValueError, 'invalid artifact'):
            bundle.verify(self.root)
        (self.root / 'manifest.json').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            bundle.verify(self.root)

    def test_escape_and_symlink_rejected(self):
        self.manifest['artifacts']['../outside'] = self.manifest['artifacts']['payload']
        self.save()
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            bundle.verify(self.root)
        del self.manifest['artifacts']['../outside']
        self.save()
        (self.root / 'link').symlink_to(self.root, target_is_directory=True)
        self.manifest['artifacts']['link/payload'] = self.manifest['artifacts']['payload']
        self.save()
        with self.assertRaisesRegex(ValueError, 'symlink'):
            bundle.verify(self.root)

    def test_existing_destination_and_failure_do_not_publish(self):
        with self.assertRaises(FileExistsError):
            bundle.build(self.root, 'default')
        self.assertEqual((self.root / 'payload').read_text(), 'original')
        destination = self.root / 'failed'
        with patch.object(bundle.subprocess, 'check_output', side_effect=OSError('injected failure')):
            with self.assertRaisesRegex(OSError, 'injected failure'):
                bundle.build(destination, 'default')
        self.assertFalse((destination / 'manifest.json').exists())
        self.assertTrue((destination / 'build.log').exists())

    def test_failed_native_build_retains_log_without_manifest(self):
        destination = self.root / 'native-failure'
        real_run = subprocess.run

        def fail_make(command, **kwargs):
            if command[0] == 'make':
                kwargs['stdout'].write('injected compiler failure\n')
                return subprocess.CompletedProcess(command, 2)
            return real_run(command, **kwargs)

        with patch.object(bundle.subprocess, 'run', side_effect=fail_make):
            with self.assertRaisesRegex(ValueError, 'make failed'):
                bundle.build(destination, 'default')
        self.assertFalse((destination / 'manifest.json').exists())
        self.assertIn('injected compiler failure', (destination / 'build.log').read_text())


@unittest.skipUnless(INSTALLED, 'supply --bundle for installed-bundle integration')
class Installed(unittest.TestCase):
    def run_command(self, command, text=None, expected=0):
        result = subprocess.run([str(p) for p in command], input=text, text=True,
                                capture_output=True, cwd=self.project, timeout=30)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result.stdout

    def test_relocated_write_rollback_restore_cli_mcp_and_c(self):
        with tempfile.TemporaryDirectory(prefix='vec-project-') as temporary:
            self.project = Path(temporary)
            local = self.project / '.sqlite-vec'
            shutil.copytree(INSTALLED, local)
            manifest = bundle.verify(local)
            self.run_command(['python3', local / 'tools/local_bundle.py', 'verify', local])
            sqlite = local / 'bin/sqlite3'
            db = self.project / 'app.db'
            self.run_command([sqlite, db], '''.bail on
CREATE TABLE tasks(id INTEGER PRIMARY KEY,title TEXT NOT NULL);
CREATE VIRTUAL TABLE ann USING vec0(v float[2] indexed by hnsw(m=4));
BEGIN;
INSERT INTO tasks VALUES(1,'kept'); INSERT INTO ann VALUES(1,'[1,0]');
SAVEPOINT s; INSERT INTO tasks VALUES(2,'discard'); INSERT INTO ann VALUES(2,'[0,0]');
ROLLBACK TO s; RELEASE s; COMMIT;
BEGIN; INSERT INTO tasks VALUES(3,'discard'); INSERT INTO ann VALUES(3,'[0,0]'); ROLLBACK;
CREATE VIRTUAL TABLE lexical USING fts5(body);
INSERT INTO lexical VALUES('kept');
.backup backup.db
''')
            self.assertEqual(self.run_command([sqlite, db,
                "SELECT title FROM tasks; SELECT rowid FROM ann WHERE v MATCH '[0,0]' AND k=1; SELECT rowid FROM lexical WHERE lexical MATCH 'kept';"]), 'kept\n1\n1\n')
            # SQLite CLI continues across statements unless .bail is enabled; prove that failure exits.
            self.run_command([sqlite, db], '.bail on\nINSERT INTO tasks VALUES(1,\'duplicate\');\n', expected=1)
            self.run_command([sqlite, self.project / 'restored.db'], '.bail on\n.restore backup.db\n')
            self.assertEqual(self.run_command([sqlite, self.project / 'restored.db',
                'PRAGMA integrity_check; SELECT title FROM tasks; SELECT count(*) FROM ann;']), 'ok\nkept\n1\n')
            actual = json.loads(self.run_command([sqlite, '-json', ':memory:',
                'SELECT sqlite_version() AS version, sqlite_source_id() AS source_id;']))[0]
            self.assertEqual(actual['version'], manifest['native_sqlite']['version'])
            self.assertEqual(actual['source_id'], manifest['native_sqlite']['source_id'])
            index_sql = 'CREATE VIRTUAL TABLE iv USING vec0(v float[2] indexed by ivf(nlist=1));'
            self.run_command([sqlite, ':memory:'], '.bail on\n' + index_sql,
                             expected=0 if manifest['features']['ivf'] else 1)
            self.adapter_journey(local, manifest)
            self.c_consumer(local, manifest)

    def adapter_journey(self, local, manifest):
        sqlite = local / 'bin/sqlite3'
        self.run_command([sqlite, self.project / 'search.db'], '''.bail on
CREATE TABLE documents(id INTEGER PRIMARY KEY,document_id TEXT,tenant TEXT,valid_from,valid_to,
 invalidated_at,source_id TEXT,body TEXT,start_ms,end_ms);
INSERT INTO documents VALUES(1,'doc:1','ours',0,NULL,NULL,'source:1','local evidence',0,100);
CREATE VIRTUAL TABLE vectors USING vec0(embedding float[2]); INSERT INTO vectors VALUES(1,'[1,0]');
CREATE VIRTUAL TABLE lexical USING fts5(body); INSERT INTO lexical(rowid,body) VALUES(1,'local evidence');
CREATE TABLE edges(src INTEGER,dst INTEGER);
''')
        extension = next((local / 'lib').glob('vec0.*'))
        config = dict(database='search.db', extension=str(extension.relative_to(self.project)), corpora=dict(demo=dict(
            documents='documents', vectors='vectors', fts='lexical', edges='edges', tenant='ours',
            as_of=100, dimensions=2)))
        path = self.project / 'config.json'
        path.write_text(json.dumps(config))
        binary = local / 'bin/sqlite-vec-search-lab'
        records = [dict(op='describe',corpus='demo'),
                   dict(op='search',corpus='demo',args=dict(text='local',vector=[1,0],limit=1)),
                   dict(op='fetch',corpus='demo',args=dict(ids=[1]))]
        output = self.run_command([binary, '--config', path, 'cli'],
                                  '\n'.join(json.dumps(r) for r in records)+'\n')
        results = [json.loads(line) for line in output.splitlines()]
        self.assertTrue(all(r['ok'] for r in results), results)
        self.assertIn(manifest['adapter_sqlite']['version'], json.dumps(results[0]))
        self.assertIn('source:1', json.dumps(results[2]))
        rejected = json.loads(self.run_command([binary,'--config',path,'cli'],
            json.dumps(dict(op='write',corpus='demo',args={}))+'\n', expected=1))
        self.assertFalse(rejected['ok'])
        # MCP discovery exercises the actual installed stdio binary without a Python SDK.
        messages = [dict(jsonrpc='2.0',id=1,method='initialize',params=dict(
            protocolVersion='2025-03-26',capabilities={},clientInfo=dict(name='bundle-test',version='1'))),
            dict(jsonrpc='2.0',method='notifications/initialized'),
            dict(jsonrpc='2.0',id=2,method='tools/list')]
        with subprocess.Popen([str(binary),'--config',str(path),'mcp'], cwd=self.project,
                              stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) as proc:
            proc.stdin.write(json.dumps(messages[0])+'\n');proc.stdin.flush()
            ready = select.select([proc.stdout],[],[],10)[0]
            if not ready:
                proc.kill()
            self.assertTrue(ready, 'MCP initialize timed out')
            initialized = json.loads(proc.stdout.readline())
            self.assertEqual(initialized['id'],1,initialized)
            proc.stdin.write('\n'.join(json.dumps(m) for m in messages[1:])+'\n');proc.stdin.flush()
            ready = select.select([proc.stdout],[],[],10)[0]
            if not ready:
                proc.kill()
            self.assertTrue(ready, 'MCP tools/list timed out')
            tools = json.loads(proc.stdout.readline())
            self.assertEqual({t['name'] for t in tools['result']['tools']},
                             {'describe','search','fetch','traverse','reach','witness'})
            proc.stdin.close()
            self.assertEqual(proc.wait(timeout=10),0,proc.stderr.read())

    def c_consumer(self, local, manifest):
        source = self.project / 'consumer.c'
        source.write_text('''#include <assert.h>
#include "sqlite3.h"
#include "sqlite-vec.h"
int main(void){sqlite3 *db=0;sqlite3_stmt *s=0;
assert(sqlite3_open(":memory:",&db)==0);assert(sqlite3_vec_init(db,0,0)==0);
assert(sqlite3_prepare_v2(db,"SELECT vec_version()",-1,&s,0)==0);
assert(sqlite3_step(s)==SQLITE_ROW);sqlite3_finalize(s);assert(sqlite3_close(db)==0);}
''')
        for extension in [local / 'lib/libsqlite_vec0.a',local / 'src/sqlite-vec.c']:
            command = ['cc','-DSQLITE_CORE','-DSQLITE_VEC_STATIC','-I'+str(local/'include'),source,
                       extension,local/'src/sqlite3.c','-lm','-o',self.project/'consumer']
            command[1:1] = shlex.split(manifest['native_cflags'])
            if bundle.sys.platform.startswith('linux'):
                command += ['-ldl','-lpthread']
            self.run_command(command)
            self.run_command([self.project / 'consumer'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path)
    args, remaining = parser.parse_known_args()
    INSTALLED = args.bundle.resolve() if args.bundle else None
    # The decorator is evaluated before parsing; select the integration class explicitly.
    Installed.__unittest_skip__ = INSTALLED is None
    unittest.main(argv=[__file__,*remaining])
