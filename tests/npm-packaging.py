"""Install scoped sqlite-dist packages offline and resolve both Node exports."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/scope_npm.py"


class ScopedNpm(unittest.TestCase):
    def test_installed_loaders_resolve_the_scoped_platform_package(self):
        with tempfile.TemporaryDirectory(prefix="sqlite-vec-npm-") as folder:
            root = Path(folder)
            source, scoped, app = root / "source", root / "scoped", root / "app"
            source.mkdir()
            app.mkdir()
            # Matches sqlite-dist's platformPackageName and resolver seams.
            platform = json.loads(subprocess.check_output(
                ["node", "-p", "JSON.stringify([process.platform,process.arch])"], text=True))
            os_name, arch = platform
            package = f"sqlite-vec-{'windows' if os_name == 'win32' else os_name}-{arch}"
            suffix = {"win32": "dll", "darwin": "dylib"}.get(os_name, "so")
            common = '''
const platform = process.platform, arch = process.arch;
function platformPackageName(platform, arch) {
  const os = platform === "win32" ? "windows" : platform;
  return `sqlite-vec-${os}-${arch}`;
}
function getLoadablePath() {
  const packageName = platformPackageName(platform, arch);
  return RESOLVE;
}
'''
            manifest = {"name": "sqlite-vec", "version": "0.1.10-alpha.4",
                        "main": "index.cjs", "module": "index.mjs",
                        "exports": {"require": "./index.cjs", "import": "./index.mjs"},
                        "optionalDependencies": {package: "0.1.10-alpha.4"}}
            fixtures = {
                "main": {"package.json": json.dumps(manifest),
                         "index.cjs": common.replace("RESOLVE", f'require.resolve(packageName + "/vec0.{suffix}")')
                                      + "module.exports = {getLoadablePath};\n",
                         "index.mjs": 'import {fileURLToPath} from "node:url";\n'
                                      + common.replace("RESOLVE", f'fileURLToPath(import.meta.resolve(packageName + "/vec0.{suffix}"))')
                                      + "export {getLoadablePath};\n"},
                "platform": {"package.json": json.dumps({"name": package, "version": "0.1.10-alpha.4"}),
                             f"vec0.{suffix}": "native fixture"},
            }
            for name, files in fixtures.items():
                with tarfile.open(source / f"{name}.tgz", "w:gz") as archive:
                    for path, content in files.items():
                        data = content.encode()
                        info = tarfile.TarInfo("package/" + path)
                        info.size = len(data)
                        archive.addfile(info, io.BytesIO(data))
            subprocess.run([sys.executable, str(SCRIPT), str(source), str(scoped),
                            "--scope", "@cercova-studios"], check=True, timeout=30)
            (app / "package.json").write_text('{"private":true}')
            subprocess.run(["npm", "install", "--offline", "--omit=optional", "--ignore-scripts",
                            "--no-audit", "--no-fund", "--cache", str(root / "empty-cache"),
                            *map(str, sorted(scoped.glob("*.tgz")))], cwd=app, check=True, timeout=60)
            check = '''
const fs = require('node:fs');
const cjs = require('@cercova-studios/sqlite-vec').getLoadablePath();
import('@cercova-studios/sqlite-vec').then(m => {
  const esm = m.getLoadablePath();
  if(cjs !== esm || fs.readFileSync(cjs, 'utf8') !== 'native fixture') process.exit(1);
});
'''
            subprocess.run(["node", "-e", check], cwd=app, check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
