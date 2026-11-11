"""Bootstrap boundary tests; real native compilation is a separate acceptance run."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
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
