"""Bootstrap boundary tests; real native compilation is a separate acceptance run."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import os
import shutil
import zipfile
from unittest.mock import MagicMock, patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/bootstrap.py'
spec = importlib.util.spec_from_file_location('bootstrap', SCRIPT)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class Bootstrap(unittest.TestCase):
    def make_process(self, returncode=0):
        process = MagicMock()
        process.returncode = returncode
        context = MagicMock()
        context.__enter__.return_value = process
        return context

    def test_usage_is_json_and_has_no_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'build'
            for args in (['dev', '--output', str(output), '--typo'], ['dev'], ['unknown']):
                result = subprocess.run([sys.executable, str(SCRIPT), *args],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)['error'], 'usage')
                self.assertEqual(result.stderr, '')
                self.assertFalse(output.exists())

    def test_missing_prerequisites_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(bootstrap, 'REPO', Path(directory)):
            with patch.object(bootstrap.shutil, 'which', return_value=None):
                state = bootstrap.inspect()
            self.assertIn('cc', state['missing'])
            self.assertIn('envsubst', state['missing'])
            self.assertIn('vendor/sqlite3.c', state['missing'])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / 'keep'
            sentinel.write_text('unchanged')
            with self.assertRaises(FileExistsError):
                bootstrap.develop(output)
            self.assertEqual(sentinel.read_text(), 'unchanged')
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_make_path_syntax_is_rejected_before_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('has space', '$(touch injected)', 'semi;colon', 'line\nbreak'):
                output = Path(directory) / name
                with self.assertRaises(ValueError):
                    bootstrap.develop(output)
                self.assertFalse(output.exists())

    def test_partial_failure_retains_log_and_allows_fresh_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'failed'
            with patch.object(bootstrap.subprocess, 'Popen', return_value=self.make_process(1)):
                with self.assertRaises(subprocess.CalledProcessError):
                    bootstrap.develop(output)
            self.assertIn('test-unit', (output / 'bootstrap.log').read_text())
            with self.assertRaises(FileExistsError):
                bootstrap.develop(output)
            success = subprocess.CompletedProcess([], 0, '1|0.0\n1\n', '')
            with patch.object(bootstrap.subprocess, 'Popen', return_value=self.make_process()), patch.object(bootstrap.subprocess, 'run', return_value=success):
                result = bootstrap.develop(Path(directory) / 'retry')
            self.assertIn('vector-knn', result['checks'])

    def test_wrong_smoke_rows_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong = subprocess.CompletedProcess([], 0, '2|1.0\n0\n', '')
            with patch.object(bootstrap.subprocess, 'Popen', return_value=self.make_process()), patch.object(bootstrap.subprocess, 'run', return_value=wrong):
                with self.assertRaisesRegex(ValueError, 'unexpected rows'):
                    bootstrap.develop(Path(directory) / 'wrong')

    def test_native_build_keeps_make_platform_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            success = subprocess.CompletedProcess([], 0, '1|0.0\n1\n', '')
            with patch.object(bootstrap.subprocess, 'Popen', return_value=self.make_process()) as make:
                with patch.object(bootstrap.subprocess, 'run', return_value=success):
                    bootstrap.develop(Path(directory) / 'build')
            args, kwargs = make.call_args
            self.assertFalse(any(arg.startswith('CFLAGS=') for arg in args[0]))
            self.assertEqual(kwargs['env']['CFLAGS'], '-DSQLITE_ENABLE_FTS5')

    def test_vendor_download_failure_stops_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            shutil.copyfile(SCRIPT.with_name('vendor.sh'), root / 'scripts/vendor.sh')
            tools = root / 'bin'
            tools.mkdir()
            for name, body in [('curl', 'exit 6'), ('unzip', 'touch unzip-was-run')]:
                tool = tools / name
                tool.write_text('#!/bin/sh\n' + body + '\n')
                tool.chmod(0o755)
            env = os.environ.copy()
            env['PATH'] = str(tools) + ':/usr/bin:/bin'
            result = subprocess.run(['bash', 'scripts/vendor.sh'], cwd=root, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 6, result.stderr)
            self.assertFalse((root / 'unzip-was-run').exists())

    def test_vendor_staging_retry_and_corrupt_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            shutil.copyfile(SCRIPT.with_name('vendor.sh'), root / 'scripts/vendor.sh')
            tools = root / 'bin'
            tools.mkdir()
            curl = tools / 'curl'
            curl.write_text('#!/bin/sh\nwhile [ "$1" != "--output" ]; do shift; done\n'
                            'cp "$VENDOR_TEST_ARCHIVE" "$2"\n')
            curl.chmod(0o755)
            archive = root / 'fixture.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('sqlite-amalgamation-3450300/sqlite3.c', 'fixture')
            temporary = root / 'downloads'
            temporary.mkdir()
            env = os.environ.copy()
            env.update(PATH=str(tools) + ':/usr/bin:/bin',
                       VENDOR_TEST_ARCHIVE=str(archive), TMPDIR=str(temporary))
            for attempt in range(3):
                if attempt == 2:
                    archive.write_text('not a ZIP')
                result = subprocess.run(['bash', str(root / 'scripts/vendor.sh')],
                                        cwd=tools, env=env, capture_output=True, text=True)
                if attempt < 2:
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                self.assertEqual((root / 'vendor/sqlite3.c').read_text(), 'fixture')
                self.assertEqual(list(temporary.iterdir()), [])

    def test_timeout_stops_the_build_group(self):
        with tempfile.TemporaryDirectory() as directory:
            context = self.make_process()
            process = context.__enter__.return_value
            process.pid = 12345
            process.wait.side_effect = [subprocess.TimeoutExpired('make', 600), 0]
            with patch.object(bootstrap.subprocess, 'Popen', return_value=context), patch.object(bootstrap.os, 'killpg') as kill:
                with self.assertRaises(subprocess.TimeoutExpired):
                    bootstrap.develop(Path(directory) / 'timeout')
            kill.assert_called_once_with(12345, bootstrap.signal.SIGKILL)


if __name__ == '__main__':
    unittest.main()
