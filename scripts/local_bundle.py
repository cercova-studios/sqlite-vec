#!/usr/bin/env python3
"""Build or verify a project-local bundle from this checkout's committed HEAD."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import string
import subprocess
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def define(header, name):
    prefix = '#define ' + name + ' '
    for line in header.read_text().splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix):].strip())
    raise ValueError(f'{name} absent from {header}')


def verify(directory):
    root = directory.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if not isinstance(manifest, dict) or manifest.get('format') != 'sqlite-vec-local-bundle/1':
        raise ValueError('unsupported bundle manifest')
    if not isinstance(manifest.get('artifacts'), dict) or not manifest['artifacts']:
        raise ValueError('unsupported or empty bundle manifest')
    for name, expected in manifest['artifacts'].items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ValueError(f'unsafe artifact path: {name}')
        target = root
        for part in path.parts:
            target /= part
            if target.is_symlink():
                raise ValueError(f'symlink artifact: {name}')
        if not isinstance(expected, dict) or not target.is_file():
            raise ValueError(f'invalid artifact: {name}')
        if target.stat().st_size != expected['bytes'] or digest(target) != expected['sha256']:
            raise ValueError(f'artifact mismatch: {name}')
    return manifest


def build(output, profile):
    if sys.platform not in ('darwin', 'linux'):
        raise ValueError('local bundles currently require macOS or Linux')
    if not (REPO / 'sqlite-vec.c').is_file():
        raise ValueError('build requires the sqlite-vec source checkout; installed tools support verify')
    # Exclusive creation reserves a fresh destination; failed builds retain their log.
    output.mkdir(parents=True)
    (output / 'build.log').touch()
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('CARGO_TARGET_', 'CARGO_PROFILE_')) or key in (
            'CFLAGS', 'CPPFLAGS', 'CXXFLAGS', 'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS',
            'CARGO_BUILD_TARGET', 'CARGO_TARGET_DIR', 'LIBSQLITE3_FLAGS',
            'SQLITE3_LIB_DIR', 'SQLITE3_INCLUDE_DIR', 'SQLITE3_STATIC', 'CC', 'AR',
        ):
            env.pop(key)
    env.update(CC='cc', AR='ar')
    native_flags = '-DSQLITE_ENABLE_FTS5'
    if profile == 'ivf':
        native_flags += ' -DSQLITE_VEC_EXPERIMENTAL_IVF_ENABLE=1'

    def capture(command, cwd=REPO):
        return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()

    revision = capture(['git', 'rev-parse', 'HEAD'])
    with tempfile.TemporaryDirectory(prefix='sqlite-vec-bundle-') as temporary:
        work = Path(temporary)
        source = work / 'source'
        source.mkdir()
        archive = work / 'source.tar'
        with archive.open('wb') as stream:
            subprocess.run(['git', 'archive', '--format=tar', revision], cwd=REPO,
                           stdout=stream, check=True)
        with tarfile.open(archive) as stream:
            stream.extractall(source, filter='data')
        (source / 'vendor').mkdir(exist_ok=True)
        for name in ['sqlite3.c', 'sqlite3.h', 'sqlite3ext.h', 'shell.c']:
            shutil.copyfile(REPO / 'vendor' / name, source / 'vendor' / name)
        version = (source / 'VERSION').read_text().strip()
        major, minor, patch = version.split('-')[0].split('.')
        header = string.Template((source / 'sqlite-vec.h.tmpl').read_text()).substitute(
            VERSION=version, VERSION_MAJOR=major, VERSION_MINOR=minor, VERSION_PATCH=patch,
            SOURCE=revision, DATE=capture(['git', 'show', '-s', '--format=%cI', revision]),
        )
        (source / 'sqlite-vec.h').write_text(header)
        inputs = [source / 'Makefile', source / 'VERSION', source / 'sqlite-vec.h.tmpl',
                  source / 'scripts/amalgamate.py', source / 'examples/sqlite3-cli/core_init.c']
        inputs += list(source.glob('sqlite-vec*.c')) + list((source / 'vendor').iterdir())
        adapter = source / 'examples/search-lab'
        inputs += [adapter / 'Cargo.toml', adapter / 'Cargo.lock'] + list((adapter / 'src').rglob('*.rs'))
        input_hashes = {str(p.relative_to(source)): digest(p) for p in sorted(inputs)}
        env['CARGO_TARGET_DIR'] = str(work / 'cargo-target')
        commands = [
            ['make', '-j2', 'loadable', 'static', 'amalgamation', 'cli'],
            ['cargo', 'build', '--offline', '--locked', '--release', '--bin',
             'sqlite-vec-search-lab', '--manifest-path', str(adapter / 'Cargo.toml')],
        ]
        with (output / 'build.log').open('w') as log:
            for command in commands:
                print(f'Building {command[0]} ({profile})', file=sys.stderr, flush=True)
                log.write(json.dumps(command) + '\n')
                log.flush()
                command_env = env | {'CFLAGS': native_flags} if command[0] == 'make' else env
                result = subprocess.run(command, cwd=source, env=command_env, stdout=log, stderr=log)
                if result.returncode:
                    raise ValueError(f'{command[0]} failed ({result.returncode}); see {output / "build.log"}')
        dist = source / 'dist'
        sqlite = json.loads(capture([str(dist / 'sqlite3'), '-json', ':memory:',
            'SELECT sqlite_version() AS version, sqlite_source_id() AS source_id, vec_debug() AS vec_debug;']))[0]
        sqlite['compile_options'] = json.loads(capture([str(dist / 'sqlite3'), '-json', ':memory:',
            'SELECT compile_options FROM pragma_compile_options ORDER BY compile_options;']))
        metadata = json.loads(capture(['cargo', 'metadata', '--offline', '--locked', '--format-version', '1',
                                      '--manifest-path', str(adapter / 'Cargo.toml')]))
        sqlite_crate = next(p for p in metadata['packages'] if p['name'] == 'libsqlite3-sys')
        bundled = Path(sqlite_crate['manifest_path']).parent / 'sqlite3'
        adapter_sqlite = dict(version=define(bundled / 'sqlite3.h', 'SQLITE_VERSION'),
                              source_id=define(bundled / 'sqlite3.h', 'SQLITE_SOURCE_ID'),
                              crate=sqlite_crate['version'], source_sha256=digest(bundled / 'sqlite3.c'))
        # Ask Make for the effective host SIMD flags, using the same fixed profile flags.
        flags_file = work / 'flags.mk'
        flags_file.write_text("bundle-flags: ; @printf '%s\\n' '$(CFLAGS)'\n")
        effective_flags = subprocess.check_output(
            ['make', '--no-print-directory', '-s', '-f', 'Makefile', '-f', str(flags_file), 'bundle-flags'],
            cwd=source, env=env | {'CFLAGS': native_flags}, text=True).strip()
        suffix = 'dylib' if sys.platform == 'darwin' else 'so'
        copies = {
            'bin/sqlite3': dist / 'sqlite3',
            'bin/sqlite-vec-search-lab': work / 'cargo-target/release/sqlite-vec-search-lab',
            f'lib/vec0.{suffix}': dist / f'vec0.{suffix}',
            'lib/libsqlite_vec0.a': dist / 'libsqlite_vec0.a',
            'include/sqlite-vec.h': source / 'sqlite-vec.h',
            'include/sqlite3.h': source / 'vendor/sqlite3.h',
            'include/sqlite3ext.h': source / 'vendor/sqlite3ext.h',
            'src/sqlite-vec.c': dist / 'sqlite-vec.c',
            'src/sqlite3.c': source / 'vendor/sqlite3.c',
            'LICENSE-MIT': source / 'LICENSE-MIT',
            'LICENSE-APACHE': source / 'LICENSE-APACHE',
            'tools/local_bundle.py': Path(__file__).resolve(),
            'README.md': REPO / 'docs/local-bundle.md',
            'docs/search-adapter.md': adapter / 'README.md',
            'docs/graph-semantics.md': adapter / 'graph-semantics.md',
            'examples/app.py': REPO / 'examples/bundle-consumer/app.py',
        }
        for name, original in copies.items():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            target.chmod(0o755 if name.startswith('bin/') else 0o644)
        artifacts = {str(p.relative_to(output)): dict(sha256=digest(p), bytes=p.stat().st_size)
                     for p in sorted(output.rglob('*')) if p.is_file()}
        enabled = sqlite['vec_debug'].split('Build flags:', 1)[1].split()
        if ('ivf' in enabled) != (profile == 'ivf'):
            raise ValueError('built IVF capability does not match the requested profile')
        manifest = dict(
            format='sqlite-vec-local-bundle/1', source_revision=revision, version=version,
            profile=profile, platform=dict(system=platform.system(), machine=platform.machine()),
            tools=dict(cc=capture(['cc', '--version']).splitlines()[0],
                       rustc=capture(['rustc', '-vV']), cargo=capture(['cargo', '--version'])),
            native_cflags=effective_flags, native_sqlite=sqlite, adapter_sqlite=adapter_sqlite,
            features=dict(fts5=True, hnsw=True, ivf='ivf' in enabled, lance=False,
                          rescore='rescore' in enabled, diskann='diskann' in enabled,
                          simd=[f for f in enabled if f in ('avx', 'neon')]),
            inputs=input_hashes, artifacts=artifacts,
        )
        # A manifest is the completion marker; partial/failed output is not a usable bundle.
        pending = output / '.manifest.tmp'
        pending.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        pending.rename(output / 'manifest.json')
        verify(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('build', help='build committed HEAD into a new directory')
    create.add_argument('--output', required=True, type=Path)
    create.add_argument('--profile', choices=['default', 'ivf'], default='default')
    check = sub.add_parser('verify', help='check installed artifact sizes and SHA256 hashes')
    check.add_argument('directory', type=Path)
    args = parser.parse_args()
    try:
        directory = (args.output if args.command == 'build' else args.directory).absolute()
        manifest = build(directory, args.profile) if args.command == 'build' else verify(directory)
        print(json.dumps(dict(ok=True, directory=str(directory), profile=manifest['profile'],
                              source_revision=manifest['source_revision'])))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f'local bundle: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
