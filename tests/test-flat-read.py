"""Sparse chunk reads retain the ordinary flat KNN filtering/transaction contract."""
import math
import sqlite3
import struct

import pytest


def connect(path=':memory:'):
    db = sqlite3.connect(path, isolation_level=None)
    db.enable_load_extension(True)
    db.load_extension('dist/vec0')
    db.enable_load_extension(False)
    return db


@pytest.mark.parametrize('kind,metric', [('float','l2'),('float','l1'),('float','cosine'),('int8','l2'),('bit','hamming')])
@pytest.mark.parametrize('text_pk', [False,True])
def test_sparse_flat_filter_oracle(kind, metric, text_pk):
    db = connect()
    dimension = 128 if kind == 'float' else 512 if kind == 'int8' else 4096
    definition = f'{kind}[{dimension}]' + (f' distance_metric={metric}' if kind == 'float' else '')
    pk = 'id text primary key,' if text_pk else ''
    name = 'id' if text_pk else 'rowid'
    convert = '?' if kind == 'float' else f'vec_{kind}(?)'
    db.execute(f'create virtual table t using vec0({pk}v {definition},p integer partition key,m integer,f float,b boolean,label text,chunk_size=1024)')
    def value(i):
        if kind == 'float':
            return struct.pack('128f', i/100, 1+i%7/10, *([0.0]*126))
        if kind == 'int8':
            return struct.pack('512b', i%121, i//121, *([0]*510))
        return i.to_bytes(2,'little') + bytes(510)
    ids = lambda i: f'key{i}' if text_pk else i-550
    db.execute('begin')
    db.executemany(f'insert into t({name},v,p,m,f,b,label) values (?,{convert},?,?,?,?,?)',
                   [(ids(i),value(i),i%2,i%17,i/10,i%3==0,'keep' if i%5 else 'drop') for i in range(1100)])
    db.commit()
    query = value(237)
    def distance(blob):
        if kind == 'bit':
            return float(sum((a^b).bit_count() for a,b in zip(blob,query)))
        fmt = '128f' if kind == 'float' else '512b'
        a,c = struct.unpack(fmt,blob),struct.unpack(fmt,query)
        if metric == 'l1':
            return sum(abs(x-y) for x,y in zip(a,c))
        if metric == 'cosine':
            return 1-sum(x*y for x,y in zip(a,c))/math.sqrt(sum(x*x for x in a)*sum(y*y for y in c))
        return math.sqrt(sum((x-y)**2 for x,y in zip(a,c)))
    selected = [1,3,17,31,255,511,601,1023,1099]
    placeholders = ','.join('?' for _ in selected)
    cases = [('',[],list(range(1100))),
             (f' and {name} in ({placeholders})',[ids(i) for i in selected],selected),
             (' and p=1 and m in (1,3,5) and f>10 and b=0 and label=?',['keep'],
              [i for i in range(1100) if i%2 and i%17 in (1,3,5) and i/10>10 and i%3!=0 and i%5]),
             (f' and p=1 and m>=0 and {name} in ({placeholders})',[ids(i) for i in selected],[i for i in selected if i%2]),
             (' and m<0',[],[])]
    for predicate,args,allowed in cases:
        for distance_filter in [False,True]:
            expected = {ids(i):distance(value(i)) for i in allowed}
            if distance_filter:
                expected = {i:d for i,d in expected.items() if d<2.0}
            sql = f'select {name},distance from t where v match {convert} and k=3'+predicate
            if distance_filter:
                sql += ' and distance<2.0'
            rows = db.execute(sql+' order by distance',[query,*args]).fetchall()
            assert len(rows) == min(3,len(expected))
            assert len({i for i,_ in rows}) == len(rows) and all(i in expected for i,_ in rows)
            assert all(math.isclose(d,expected[i],rel_tol=2e-5,abs_tol=2e-6) for i,d in rows)
            if rows:
                cutoff = sorted(expected.values())[len(rows)-1]
                assert max(expected[i] for i,_ in rows) <= cutoff+2e-6
    db.close()


def test_sparse_flat_pending_writes_and_rollbacks(tmp_path):
    path = tmp_path/'sparse.db'
    db = connect(path)
    db.execute('create virtual table t using vec0(v float[128],w float[128],p integer partition key,m integer,chunk_size=1024)')
    def vec(x):
        return struct.pack('128f',x,*([0.0]*127))
    live = {i:float(i) for i in range(1100)}
    db.execute('begin')
    db.executemany('insert into t(rowid,v,w,p,m) values (?,?,?,?,?)',[(i,vec(i),vec(-i),i%2,i%17) for i in live])
    db.commit()
    def check():
        selected = [1,3,511,1099,1201]
        allowed = [i for i in selected if i in live and i%2 and i%17>=0]
        for col,sign in [('v',1),('w',-1)]:
            rows = db.execute(f'select rowid,distance from t where {col} match ? and k=3 and p=1 and m>=0 and rowid in (1,3,511,1099,1201) order by distance',[vec(.37)]).fetchall()
            expected = sorted(allowed,key=lambda i:abs(sign*live[i]-.37))[:3]
            assert [i for i,_ in rows] == expected
            assert all(math.isclose(d,abs(sign*live[i]-.37),rel_tol=2e-6) for i,d in rows)
    check()
    db.execute('begin')
    db.execute('insert into t(rowid,v,w,p,m) values (1201,?,?,1,11)',[vec(2),vec(-2)])
    live[1201] = 2.
    db.execute('update t set v=?,w=? where rowid=1099',[vec(4),vec(-4)])
    live[1099] = 4.
    check()  # Flushes pending vector ranges before selective KNN reads.
    db.execute('savepoint s')
    old = live.copy()
    db.execute('delete from t where rowid=1'); live.pop(1)
    db.execute('update t set v=?,w=? where rowid=511',[vec(0),vec(0)]); live[511] = 0.
    check()
    db.execute('rollback to s'); live = old
    check()
    db.execute('release s')
    with pytest.raises(sqlite3.Error):
        db.execute('insert into t(rowid,v,w,p,m) values (1301,?,?,1,1),(1,?,?,1,1)',[vec(0),vec(0),vec(0),vec(0)])
    assert db.execute('select rowid from t where rowid=1301').fetchone() is None
    check()
    db.execute('rollback'); live = {i:float(i) for i in range(1100)}
    check()
    db.execute('begin')
    db.execute('delete from t where rowid=1'); live.pop(1)
    db.execute('insert into t(rowid,v,w,p,m) values (1201,?,?,1,11)',[vec(2),vec(-2)]); live[1201] = 2.
    db.commit(); db.close()
    db = connect(path)
    check()
    db.close()


def test_sparse_flat_flush_failure_and_retry():
    db = connect()
    db.execute('create virtual table t using vec0(v float[128],chunk_size=1024)')
    value = struct.pack('128f',1,*([0.0]*127))
    db.execute('begin')
    db.execute('insert into t(rowid,v) values (511,?)',[value])
    sql = 'select rowid,distance from t where v match ? and k=3 and rowid in (511) order by distance'
    db.execute('pragma query_only=on')
    with pytest.raises(sqlite3.Error):
        db.execute(sql,[value]).fetchall()
    db.execute('pragma query_only=off')
    assert db.in_transaction
    assert db.execute(sql,[value]).fetchall() == [(511,0.0)]
    db.execute('rollback')
    assert db.execute(sql,[value]).fetchall() == []
    db.close()


def test_sparse_flat_blob_shape_failure_and_recovery():
    db = connect()
    db.execute('create virtual table t using vec0(v float[128],chunk_size=1024)')
    value = struct.pack('128f',1,*([0.0]*127))
    db.execute('insert into t(rowid,v) values (7,?)',[value])
    sql = 'select rowid,distance from t where v match ? and k=3 and rowid in (7) order by distance'
    assert db.execute(sql,[value]).fetchall() == [(7,0.0)]
    original = db.execute('select vectors from t_vector_chunks00').fetchone()[0]
    db.execute('update t_vector_chunks00 set vectors=zeroblob(512)')
    with pytest.raises(sqlite3.Error, match="vectors blob size doesn't match"):
        db.execute(sql,[value]).fetchall()
    db.execute('update t_vector_chunks00 set vectors=?',[original])
    assert db.execute(sql,[value]).fetchall() == [(7,0.0)]
    db.close()
