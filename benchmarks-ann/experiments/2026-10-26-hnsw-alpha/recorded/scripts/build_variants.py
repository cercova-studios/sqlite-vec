from pathlib import Path
import subprocess,hashlib,json,shutil
root=Path('/Users/rohit/sandbox/sqlite-vec');out=Path('/tmp/sqlite-vec-alpha-experiment')
flags=['-fPIC','-shared','-Wall','-Wextra','-O3','-mcpu=apple-m1','-DSQLITE_VEC_ENABLE_NEON']
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
manifest={'head':head,'working_diff':subprocess.check_output(['git','diff','--no-ext-diff'],cwd=root,text=True),'flags':flags,'compiler':subprocess.check_output(['cc','--version'],text=True),'variants':{}}
for alpha in [1.0,1.05,1.1,1.2]:
    name=f'alpha-{alpha:g}';folder=out/'builds'/name;folder.mkdir(parents=True,exist_ok=True)
    hashes={}
    for source in root.glob('sqlite-vec*'):
        if source.is_file():
            shutil.copy2(source,folder/source.name)
            hashes[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
    target=folder/'sqlite-vec-hnsw.c';text=target.read_text()
    needle='if (x->dist(v, keptVec[j], &x->dims) < cands[i].d)'
    assert text.count(needle)==1
    if alpha!=1.0:text=text.replace(needle,f'if ({alpha:g}f * x->dist(v, keptVec[j], &x->dims) < cands[i].d)')
    target.write_text(text)
    cmd=['cc',*flags,'-I'+str(root/'vendor'),str(folder/'sqlite-vec.c'),'-o',str(folder/'vec0.dylib')]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    manifest['variants'][name]={'alpha':alpha,'library':str(folder/'vec0.dylib'),'library_sha256':hashlib.sha256((folder/'vec0.dylib').read_bytes()).hexdigest(),'hnsw_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'source_hashes':hashes,'command':cmd,'build_stderr':result.stderr}
    print(name,'built',flush=True)
(out/'builds/manifest.json').write_text(json.dumps(manifest,indent=2))
