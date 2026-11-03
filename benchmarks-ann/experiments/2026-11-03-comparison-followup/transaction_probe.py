"""Tiny same-connection vector-index lifecycle probes; unsupported is not passed."""
from pathlib import Path
import importlib,importlib.metadata,json,sqlite3,tempfile
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'2026-11-03-engine-comparison'
def run(engine):
    result={'engine':engine,'stages':[],'passed':False};stage='open'
    try:
        with tempfile.TemporaryDirectory(prefix='vector-lifecycle-') as root:
            path=Path(root)/'db.sqlite'
            def connect():
                if engine=='sqlite':
                    db=sqlite3.connect(path,isolation_level=None);db.enable_load_extension(True);db.load_extension(str(PREVIOUS/'.work/build/vec0.dylib'));db.enable_load_extension(False);return db
                return importlib.import_module(engine).connect(str(path))
            db=connect()
            if engine=='sqlite':
                db.execute('create virtual table docs using vec0(v float[2] indexed by hnsw(m=4,ef_construction=16))')
                insert=lambda i:db.execute('insert into docs(rowid,v) values (?,?)',[i,json.dumps([float(i),0.])])
                query=lambda:{r[0] for r in db.execute("select rowid from docs where v match '[1,0]' and k=10 order by distance").fetchall()}
            else:
                db.execute('create table docs(id integer primary key,v F32_BLOB(2))')
                insert=lambda i:db.execute('insert into docs values (?,vector32(?))',[i,json.dumps([float(i),0.])])
                if engine=='libsql':db.execute("create index vi on docs(libsql_vector_idx(v,'metric=l2'))")
                query=(lambda:{r[0] for r in db.execute("select * from vector_top_k('vi',vector32('[1,0]'),10)").fetchall()}) if engine=='libsql' else (lambda:{r[0] for r in db.execute("select id from docs order by vector_distance_l2(v,vector32('[1,0]')) limit 10").fetchall()})
            insert(1);db.commit();assert query()=={1}
            stage='own writes';db.execute('begin');insert(2);assert query()=={1,2};result['stages'].append(stage)
            stage='savepoint';db.execute('savepoint s');insert(3);assert query()=={1,2,3};db.execute('rollback to s');assert query()=={1,2};db.execute('release s');result['stages'].append(stage)
            stage='full rollback';db.rollback();assert query()=={1};result['stages'].append(stage)
            stage='commit/reopen';db.execute('begin');insert(2);db.commit();db.close();db=connect();assert query()=={1,2};db.close();result['stages'].append(stage)
            result['passed']=True
    except Exception as e:result.update(failed_stage=stage,error=type(e).__name__+': '+str(e))
    return result
if __name__=='__main__':
    result=[run(e) for e in ['sqlite','libsql','turso']]
    (HERE/'recorded/transaction-probes.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
