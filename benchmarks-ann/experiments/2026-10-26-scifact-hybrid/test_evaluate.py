"""Calibrate metrics, query escaping and lifecycle checks before evaluation."""
from pathlib import Path
import json
import math
import tempfile
import unittest
import numpy as np
import evaluate as ev


class EvaluationChecks(unittest.TestCase):
    def test_known_metrics_and_missing_judgments(self):
        result=ev.metrics([2,1],{'1':1,'3':1})
        self.assertEqual(result['recall_at_10'],.5)
        self.assertEqual(result['reciprocal_rank'],.5)
        self.assertAlmostEqual(result['ndcg_at_10'],(1/math.log2(3))/(1+1/math.log2(3)))
        self.assertIsNone(ev.metrics([2],{})['recall_at_10'])
        self.assertEqual(ev.metrics([],{'1':1})['ndcg_at_10'],0)
        with self.assertRaises(AssertionError):ev.metrics([1,1],{'1':1})

    def test_graded_linear_gain(self):
        value=ev.metrics([2,1],{'1':3,'2':1})['ndcg_at_10']
        self.assertAlmostEqual(value,(1+3/math.log2(3))/(3+1/math.log2(3)))

    def test_scope_boundaries(self):
        self.assertTrue(ev.allowed(('ours',100,101,None)))
        for attributes in [('other',0,None,None),('ours',101,None,None),('ours',0,100,None),('ours',0,None,100)]:
            self.assertFalse(ev.allowed(attributes))
        self.assertFalse(ev.allowed(ev.scope(42,'invalidated',{42})))
        self.assertTrue(ev.allowed(ev.scope(42,'full',set())))

    def test_fts_operators_are_literal_tokens(self):
        expression=ev.lexical_queries([{'_id':'q','text':'NF-κB "activation" OR 3.14'}])['q']
        self.assertIn('"or"',expression)
        self.assertIn('"activation"',expression)
        db=ev.sqlite3.connect(':memory:');db.execute('create virtual table f using fts5(body)')
        db.execute('insert into f values (?)',('activation',))
        self.assertEqual(db.execute('select count(*) from f where f match ?',(expression,)).fetchone()[0],1)
        db.close()

    def test_real_embedding_pending_invalidation_rollback_and_reopen(self):
        work=ev.HERE/'.work';data=json.loads((work/'data.json').read_text())
        documents=data['documents'][:12];vectors=np.load(work/'documents.npy')[:12]
        target=int(documents[0]['_id']);ids={int(d['_id']) for d in documents}
        with tempfile.TemporaryDirectory(prefix='scifact-lifecycle-') as directory:
            path=Path(directory)/'test.db';extension=work/'build/vec0.dylib'
            db=ev.connect(path,extension);ev.setup(db,documents,vectors)
            expression=ev.lexical_queries([{'_id':'q','text':documents[0]['title']}])['q']
            p=dict(tenant='ours',as_of=100,q_vec=vectors[0].tobytes(),q_text=expression,branch_limit=10)
            sqls=[ev.VECTOR,ev.KEYWORD,ev.HYBRID]
            before=[db.execute(sql,p).fetchall() for sql in sqls]
            self.assertEqual(before[0][0][0],target)
            db.execute('savepoint trial');db.execute('update doc_scope set invalidated_at=100 where id=?',(target,))
            for sql in sqls:self.assertNotIn(target,[r[0] for r in db.execute(sql,p)])
            db.execute('rollback to trial');db.execute('release trial')
            self.assertEqual([db.execute(sql,p).fetchall() for sql in sqls],before)
            db.execute('begin');db.execute('update doc_scope set invalidated_at=100 where id=?',(target,));db.execute('rollback')
            self.assertEqual([db.execute(sql,p).fetchall() for sql in sqls],before)
            db.execute('begin');db.execute('update doc_scope set invalidated_at=100 where id=?',(target,));db.execute('commit')
            after=[db.execute(sql,p).fetchall() for sql in sqls]
            for rows in after:self.assertNotIn(target,[r[0] for r in rows])
            db.close();db=ev.connect(path,extension)
            self.assertEqual([db.execute(sql,p).fetchall() for sql in sqls],after)
            self.assertEqual(db.execute('select count(*) from vec_docs').fetchone()[0],len(ids))
            db.close()


if __name__=='__main__':
    unittest.main()
