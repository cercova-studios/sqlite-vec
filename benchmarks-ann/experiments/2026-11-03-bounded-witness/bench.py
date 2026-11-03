"""Build isolated libraries and compare bounded witnesses in alternating processes."""
from pathlib import Path
from collections import deque
import hashlib, json, platform, random, shutil, sqlite3, statistics, subprocess, sys, time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = 'c956253'
WORK = HERE / '.work'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def graphs():
    rng = random.Random(42)
    return {
        'chain': ([(i,i,i+1) for i in range(4999)], [(4970,4999),(4980,4999),(4990,4999),(0,4999)]),
        'star': ([(i,0,i+1) for i in range(4999)], [(0,4999),(4999,0)]),
        'random': ([(i,rng.randrange(1000),rng.randrange(1000)) for i in range(4000)], [(0,999),(100,500),(500,100),(999,0)]),
        'parallel': ([(i,1,2) for i in range(19999)], [(1,2),(1,1)]),
    }

def oracle(edges, source, target):
    adj = {}
    for edge,s,d in edges:
        adj.setdefault(s, []).append((d,edge))
    # Compute distances independently of the native representation.
    distance = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v,_ in adj.get(u, []):
            if v not in distance:
                distance[v] = distance[u] + 1
                queue.append(v)
    return distance.get(target)

def query(db, table, source, target, work=10**9, memory=16*1024*1024):
    return db.execute('select graph_witness(?,\'src\',\'dst\',\'id\',?,?,?,?)', [table,source,target,work,memory]).fetchone()[0]

def child(variant, memory_probe=False):
    db = sqlite3.connect(WORK / 'graph.db', isolation_level=None)
    db.enable_load_extension(True)
    db.load_extension(str(WORK / variant / 'vec0.dylib'))
    db.enable_load_extension(False)
    if memory_probe:
        result = {}
        for name,source,target in [('chain',0,4999),('random',0,999),('parallel',1,2)]:
            probes = []
            # Allocator rounding can reject arbitrary sizes; test a fixed grid
            # rather than assume every byte-budget success threshold is monotone.
            for exponent in range(7,25):
                memory = 1 << exponent
                try:
                    query(db,name,source,target,memory=memory)
                except sqlite3.DataError as e:
                    assert 'memory limit' in str(e)
                    passed = False
                else:
                    passed = True
                probes.append(dict(bytes=memory,passed=passed))
            assert any(p['passed'] for p in probes)
            result[name] = probes
        db.close()
        print(json.dumps(result))
        return
    points = []
    for name,(edges,pairs) in graphs().items():
        edge_set = set(edges)
        for source,target in pairs:
            start = time.perf_counter_ns()
            raw = query(db,name,source,target)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            path = json.loads(raw)
            hops = oracle(edges,source,target)
            if hops is None:
                assert path == []
            else:
                assert len(path) == hops+1 and path[0] == [source,None] and path[-1][0] == target
                assert all((b[1],a[0],b[0]) in edge_set for a,b in zip(path,path[1:]))
            points.append(dict(graph=name,source=source,target=target,ms=elapsed,path_nodes=len(path),path_sha256=sha(raw.encode())))
    try:
        query(db,'chain',0,4999,work=100000)
    except sqlite3.Error as e:
        budget = str(e)
        assert variant == 'before' and 'work limit' in budget
    else:
        budget = 'passed'
        assert variant == 'after'
    db.close()
    print(json.dumps(dict(variant=variant,points=points,work_100000=budget)))

def build():
    WORK.mkdir(exist_ok=True)
    sources = subprocess.check_output(['git','ls-tree','--name-only',BASE],cwd=REPO,text=True).splitlines()
    sources = [n for n in sources if n.startswith('sqlite-vec') and n.endswith('.c')]
    manifests = {}
    for variant in ['before','after']:
        folder = WORK / variant
        folder.mkdir(exist_ok=True)
        for name in sources:
            data = (REPO/name).read_bytes() if variant == 'after' and name == 'sqlite-vec-graph.c' else subprocess.check_output(['git','show',BASE+':'+name],cwd=REPO)
            (folder/name).write_bytes(data)
        shutil.copyfile(REPO/'sqlite-vec.h',folder/'sqlite-vec.h')
        command = ['cc','-O3','-fPIC','-shared','-mcpu=apple-m1','-DSQLITE_VEC_ENABLE_NEON','-I'+str(REPO/'vendor'),str(folder/'sqlite-vec.c'),'-o',str(folder/'vec0.dylib')]
        subprocess.run(command,check=True)
        manifests[variant] = {p.name:sha(p.read_bytes()) for p in folder.iterdir() if p.suffix in ['.c','.h','.dylib']}
    return manifests

def summary(runs):
    result = []
    for name,(_,pairs) in graphs().items():
        for source,target in pairs:
            both = {}
            paths = set()
            for variant in ['before','after']:
                points = [p for r in runs if r['variant'] == variant for p in r['points'] if (p['graph'],p['source'],p['target']) == (name,source,target)]
                assert len(points) == 5
                paths.update(p['path_sha256'] for p in points)
                both[variant] = statistics.median(p['ms'] for p in points)
            assert len(paths) == 1
            result.append(dict(graph=name,source=source,target=target,**both,speedup=both['before']/both['after']))
    return result

def main():
    if len(sys.argv)>1:
        child(sys.argv[1], len(sys.argv)>2)
        return
    manifests = build()
    path = WORK / 'graph.db'
    if path.exists():
        path.unlink()  # Only this benchmark's disposable generated database.
    db = sqlite3.connect(path)
    for name,(edges,_) in graphs().items():
        db.execute(f'create table {name}(id integer primary key,src integer,dst integer)')
        db.executemany(f'insert into {name} values (?,?,?)',edges)
    db.commit()
    db.close()
    runs = []
    for repeat in range(5):
        for variant in (['before','after'] if repeat % 2 == 0 else ['after','before']):
            r = subprocess.run([sys.executable,__file__,variant],check=True,capture_output=True,text=True)
            result = json.loads(r.stdout)
            result['round'] = repeat
            runs.append(result)
    memory = {}
    for variant in ['before','after']:
        r = subprocess.run([sys.executable,__file__,variant,'memory'],check=True,capture_output=True,text=True)
        memory[variant] = json.loads(r.stdout)
    result = dict(baseline=BASE,source_sha256=sha(Path(__file__).read_bytes()),platform=platform.platform(),python=sys.version,
                  sqlite_version=sqlite3.sqlite_version,compiler=subprocess.check_output(['cc','--version'],text=True).splitlines()[0],
                  libraries=manifests,graphs_sha256=sha(json.dumps(graphs()).encode()),runs=runs,summary=summary(runs),payload_budget_probes=memory)
    (HERE/'recorded/results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))

if __name__ == '__main__':
    main()
