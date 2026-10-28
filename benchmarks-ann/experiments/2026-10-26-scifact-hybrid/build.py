"""Build a benchmark library from a single committed C snapshot."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess


def main():
    root=Path(__file__).resolve().parents[3]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=Path(__file__).resolve().parent/'.work')
    args=parser.parse_args();out=args.work/'build';out.mkdir(parents=True,exist_ok=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    names=subprocess.check_output(['git','ls-files','sqlite-vec*.c'],cwd=root,text=True).splitlines()
    for name in names:
        (out/name).write_bytes(subprocess.check_output(['git','show',f'{head}:{name}'],cwd=root))
    (out/'sqlite-vec.h').write_bytes((root/'sqlite-vec.h').read_bytes())
    command=['cc','-fPIC','-shared','-Wall','-Wextra','-O3','-mcpu=apple-m1','-DSQLITE_VEC_ENABLE_NEON',
             '-I'+str(root/'vendor'),str(out/'sqlite-vec.c'),'-o',str(out/'vec0.dylib')]
    result=subprocess.run(command,capture_output=True,text=True,check=True)
    manifest=dict(head=head,command=command,compiler=subprocess.check_output(['cc','--version'],text=True),
        stderr=result.stderr,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('sqlite-vec*')},
        excluded_working_diff=subprocess.check_output(['git','diff','--no-ext-diff','--',*names],cwd=root,text=True),
        extension_sha256=hashlib.sha256((out/'vec0.dylib').read_bytes()).hexdigest())
    (args.work/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Built committed snapshot',head,flush=True)
    print(result.stderr,flush=True)


if __name__=='__main__':
    main()
