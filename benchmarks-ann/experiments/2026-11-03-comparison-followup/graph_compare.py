"""Compare equivalent minimum-hop sets and one shortest witness, not path bags."""
from pathlib import Path
from collections import deque
import csv,gzip,hashlib,importlib.metadata,importlib.util,json,resource,sqlite3,sys,tempfile,time
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
PREVIOUS=HERE.parent/'2026-11-03-engine-comparison'
def bfs(edges,start,depth):
    adj={}
    for _,s,d in edges:
        if s is not None and d is not None:adj.setdefault(s,[]).append(d)
    seen={start:0};queue=deque([start])
    while queue:
        u=queue.popleft()
        if seen[u]==depth:continue
        for v in adj.get(u,[]):
            if v not in seen:seen[v]=seen[u]+1;queue.append(v)
    return seen
class Graph:
    def __init__(self,engine,root,deep=False):
        self.engine=engine;self.root=root
        if engine=='sqlite':
            self.db=sqlite3.connect(root/'graph.db',isolation_level=None);self.db.enable_load_extension(True);self.db.load_extension(str(PREVIOUS/'.work/build/vec0.dylib'));self.db.enable_load_extension(False)
            self.version=self.db.execute('select vec_version()').fetchone()[0]
        else:
            import ladybug as lb
            self.db=lb.Database(root/'graph.lbug',max_num_threads=1,buffer_pool_size=64*1024*1024);self.c=lb.Connection(self.db);self.c.execute('CALL var_length_extend_max_depth='+str(5000 if deep else 30));self.version=importlib.metadata.version('ladybug')
    def load(self,name,nodes,edges):
        if self.engine=='sqlite':
            self.db.execute(f'create table {name}(id integer primary key,src integer,dst integer)')
            self.db.execute('begin');self.db.executemany(f'insert into {name} values (?,?,?)',edges);self.db.execute('commit')
        else:
            self.c.execute(f'CREATE NODE TABLE N{name}(id INT64, PRIMARY KEY(id))')
            self.c.execute(f'CREATE REL TABLE E{name}(FROM N{name} TO N{name}, eid INT64)')
            path=self.root/(name+'-nodes.csv')
            with path.open('w') as f:
                w=csv.writer(f);w.writerows((n,) for n in sorted(set(nodes)|{n for _,s,d in edges for n in [s,d] if n is not None}))
            self.c.execute(f"COPY N{name} FROM '{path}' (HEADER=false)")
            path=self.root/(name+'-edges.csv')
            with path.open('w') as f:csv.writer(f).writerows((s,d,i) for i,s,d in edges if s is not None and d is not None)
            self.c.execute(f"COPY E{name} FROM '{path}' (HEADER=false)")
    def reach(self,name,start,depth):
        if self.engine=='sqlite':return dict(self.db.execute(f"select node,depth from graph_bfs('{name}','src','dst',?,?)",[start,depth]))
        if depth==0:return {start:0}
        depth=depth if depth is not None else 30
        rows=self.c.execute(f'MATCH (a:N{name} {{id:$start}})-[p:E{name}* SHORTEST 1..{depth}]->(b:N{name}) RETURN b.id,min(length(p))',{'start':start}).get_all()
        result={start:0}
        for node,hops in rows:result[node]=min(result.get(node,hops),hops)
        return result
    def path(self,name,source,target,bounded=False,work=1000000000):
        if self.engine=='sqlite':
            if bounded:return json.loads(self.db.execute(f"select graph_witness('{name}','src','dst','id',?,?,?,16777216)",[source,target,work]).fetchone()[0])
            return [[n,None] for _,n in self.db.execute(f"select step,node from graph_shortest_path('{name}','src','dst',?,?)",[source,target])]
        if source==target:return [[source,None]]
        rows=self.c.execute(f'MATCH (a:N{name} {{id:$source}})-[p:E{name}* SHORTEST]->(b:N{name} {{id:$target}}) RETURN nodes(p),rels(p)',{'source':source,'target':target}).get_all()
        if not rows:return []
        middle,rels=rows[0];nodes=[source]+[n['id'] for n in middle]+[target]
        assert len(nodes)==len(rels)+1
        return [[nodes[0],None]]+[[n,r['eid']] for n,r in zip(nodes[1:],rels)]
    def close(self):
        if self.engine!='sqlite':self.c.close()
        self.db.close()
def main(engine,deep=False):
    result={'engine':engine,'g0':[],'timings':[],'setup':[],'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    spec=importlib.util.spec_from_file_location('g0_oracle',REPO/'examples/search-lab/graph_oracle.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    with tempfile.TemporaryDirectory(prefix='graph-compare-') as root:
        g=Graph(engine,Path(root),deep);result['version']=g.version
        fixtures=json.loads((REPO/'examples/search-lab/graph-fixtures.json').read_text())['graphs']
        for i,graph in enumerate(fixtures):
            name='g'+str(i);g.load(name,graph['nodes'],graph['edges'])
            for case in graph['cases']:
                got=g.reach(name,case['start'],case['depth']);expected=oracle.minimum_hops([(s,d) for _,s,d in graph['edges']],[case['start']],case['depth']);assert got==expected,(graph['name'],case,got,expected)
                result['g0'].append({'graph':graph['name'],'start':case['start'],'depth':case['depth'],'nodes':len(got)})
        rng=np.random.default_rng(42);n=5000
        edges=[(i,i,i+1) for i in range(n-1)];start=time.perf_counter();g.load('chain',range(n),edges);result['setup'].append({'graph':'chain','seconds':time.perf_counter()-start})
        distance={(source,n-1):n-1-source for source in ([0,2500,4990] if deep else [4970,4980,4990])}
        edge_set={(s,d):i for i,s,d in edges}
        for source,target in distance:
            modes=['legacy','bounded'] if engine=='sqlite' else ['shortest']
            for mode in modes:
                for repeat in range(3):
                    begin=time.perf_counter_ns();path=g.path('chain',source,target,mode=='bounded');elapsed=(time.perf_counter_ns()-begin)/1e6
                    assert len(path)==distance[source,target]+1 and path[0][0]==source and path[-1][0]==target
                    for a,b in zip(path,path[1:]):
                        assert (a[0],b[0]) in edge_set
                        if mode!='legacy':assert b[1]==edge_set[a[0],b[0]]
                    result['timings'].append({'graph':'chain','mode':mode,'source':source,'target':target,'round':repeat,'ms':elapsed,'path_nodes':len(path)})
        edges=[(i,int(s),int(d)) for i,(s,d) in enumerate(rng.integers(0,n,size=(10000,2)))];start=time.perf_counter();g.load('random',range(n),edges);result['setup'].append({'graph':'random','seconds':time.perf_counter()-start})
        starts=rng.choice(n,30,replace=False).tolist();expected={s:bfs(edges,s,3) for s in starts}
        for s in starts:assert g.reach('random',s,3)==expected[s]
        for repeat in range(3):
            for s in starts:
                begin=time.perf_counter_ns();got=g.reach('random',s,3);elapsed=(time.perf_counter_ns()-begin)/1e6;assert got==expected[s]
                result['timings'].append({'graph':'random','mode':'bfs3','source':s,'round':repeat,'ms':elapsed,'nodes':len(got)})
        if engine=='sqlite':
            try:g.path('chain',0,n-1,True,work=1)
            except sqlite3.Error as e:result['work_failure']=str(e)
            else:raise AssertionError('work limit ignored')
            assert g.path('chain',4990,n-1,True)[-1][0]==n-1
        g.close();result['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (HERE/'recorded'/(f'graph-{engine}'+('-deep-probe.json' if deep else '.json'))).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    engine=sys.argv[1];deep='--deep' in sys.argv
    try:main(engine,deep)
    except RuntimeError as e:
        if not (engine=='ladybug' and deep):raise
        (HERE/'recorded/graph-ladybug-deep-probe.json').write_text(json.dumps({'engine':engine,'status':'failed','error':str(e),'buffer_pool_bytes':67108864,'depth':5000,'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
        print('Deep Ladybug probe failed:',str(e),file=sys.stderr)
