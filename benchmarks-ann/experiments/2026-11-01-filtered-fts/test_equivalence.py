"""Check integer membership, ranking and transactional parity with the prior SQL."""
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import benchmark as b


class QueryEquivalence(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='fts-equivalence-')
        self.path=Path(self.temp.name)/'test.db';self.library=b.HERE/'.work/build/vec0.dylib'
        self.db=b.s.connect(self.path,self.library)
        self.ids=[-(2**63),-1000000000,-1,0,1,2,17,1000000000,2**63-1]+list(range(100,150))
        docs=[dict(_id=str(i),title='deploy procedure',text=('credential backup' if n%2 else 'credential restore')) for n,i in enumerate(self.ids)]
        b.s.setup(self.db,docs,np.array([[n,0] for n in range(len(docs))],dtype=np.float32))
        rng=random.Random(42)
        for i in self.ids:
            self.db.execute('update doc_scope set tenant=?,valid_from=?,valid_to=?,invalidated_at=? where id=?',
                            (rng.choice(['ours','other']),rng.choice([0,100,101]),rng.choice([None,100,101]),rng.choice([None,None,100]),i))
        self.variants=b.sqls()

    def tearDown(self):
        self.db.close();self.temp.cleanup()

    def compare(self,p):
        outputs={}
        for method,variants in self.variants.items():
            old=self.db.execute(variants['baseline'],p).fetchall()
            new=self.db.execute(variants['candidate'],p).fetchall()
            self.assertEqual(new,old,(method,p))
            valid={r[0] for r in self.db.execute('''select id from doc_scope where tenant=?
                and invalidated_at is null and valid_from<=? and (valid_to is null or valid_to>?)''',
                (p['tenant'],p['as_of'],p['as_of']))}
            self.assertTrue({r[0] for r in new}<=valid)
            outputs[method]=new
        return outputs

    def test_scopes_operators_limits_and_signed_rowids(self):
        texts=['deploy','credential OR backup','"deploy procedure"','depl*','deploy NOT backup',
               'NEAR(deploy credential, 3)','absenttoken']
        for tenant in ('ours','other','absent'):
            for as_of in (99,100,101):
                for limit in (1,2,50):
                    for text in texts:
                        self.compare(dict(tenant=tenant,as_of=as_of,q_vec='[0,0]',q_text=text,branch_limit=limit))
        # All edge rowids must participate when every row is eligible.
        self.db.execute("update doc_scope set tenant='ours',valid_from=0,valid_to=null,invalidated_at=null")
        p=dict(tenant='ours',as_of=100,q_vec='[0,0]',q_text='deploy',branch_limit=50)
        result=self.compare(p)
        self.assertEqual([r[0] for r in result['keyword']],sorted(self.ids)[:50])
        self.assertTrue(any(r[0]==-(2**63) for r in result['hybrid']))
        self.db.execute('pragma automatic_index=off')
        self.compare(p)

    def test_pending_writes_savepoints_rollback_commit_reopen(self):
        self.db.execute("update doc_scope set tenant='ours',valid_from=0,valid_to=null,invalidated_at=null")
        p=dict(tenant='ours',as_of=100,q_vec='[0,0]',q_text='deploy',branch_limit=50)
        before=self.compare(p)
        def mutate():
            self.db.execute('update doc_scope set invalidated_at=100 where id=?',(self.ids[0],))
            self.db.execute("update fts_docs set body='unrelated' where rowid=?",(self.ids[1],))
            self.db.execute("update vec_docs set embedding='[500,0]' where rowid=?",(self.ids[1],))
        self.db.execute('savepoint trial');mutate();self.compare(p)
        self.db.execute('rollback to trial');self.db.execute('release trial');self.assertEqual(self.compare(p),before)
        self.db.execute('begin');mutate();self.compare(p);self.db.execute('rollback');self.assertEqual(self.compare(p),before)
        self.db.execute('begin');mutate();after=self.compare(p);self.db.execute('commit')
        self.db.close();self.db=b.s.connect(self.path,self.library)
        self.assertEqual(self.compare(p),after)
        self.assertNotIn(self.ids[0],[r[0] for r in after['hybrid']])
        self.db.execute('analyze');self.assertEqual(self.compare(p),after)


if __name__=='__main__':
    unittest.main()
