"""Lifecycle and deliberately unsafe-query checks for the evaluation oracle."""
from pathlib import Path
import copy
import os
import tempfile
import unittest
import evaluate as ev

EXTENSION = Path(os.environ.get('SCOPED_HYBRID_EXTENSION', ev.REPO/'dist/vec0.dylib')).resolve()


class EvaluationChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='hybrid-evaluation-test-')
        self.path = Path(self.directory.name)/'test.db'
        self.documents = ev.corpus(200,0)
        self.db = ev.connect(self.path,EXTENSION)
        ev.setup(self.db,self.documents)

    def tearDown(self):
        self.db.close()
        self.directory.cleanup()

    def answers(self):
        return {(q['name'],method):ev.retrieve(self.db,method,q,10)
                for q in ev.FIXTURE['queries'] for method in ('vector','keyword','hybrid')}

    def assert_contents(self):
        for doc in self.documents.values():
            self.assertEqual(self.db.execute('select body from fts_docs where rowid=?',(doc['id'],)).fetchone()[0],doc['body'])
            value=self.db.execute('select embedding from vec_docs where rowid=?',(doc['id'],)).fetchone()[0]
            self.assertEqual(value,ev.blob(doc['vector']))
            scope=self.db.execute('select tenant,valid_from,valid_to,invalidated_at,source_id from doc_scope where id=?',(doc['id'],)).fetchone()
            self.assertEqual(scope,tuple(doc[k] for k in ('tenant','valid_from','valid_to','invalidated_at','source_id')))

    def assert_queries(self):
        self.assert_contents()
        for query in ev.FIXTURE['queries']:
            for limit in (10,50):
                ev.check_vector(self.db,self.documents,query,limit)
                ev.check_fusion(self.db,self.documents,query,limit)
                for method in ('vector','keyword','hybrid'):
                    result=ev.assess(ev.retrieve(self.db,method,query,limit),self.documents,query)
                    self.assertEqual(result['ineligible_ids'],[])
                    self.assertEqual(result['stale_ids'],[])

    def change(self):
        self.db.execute('update doc_scope set invalidated_at=100 where id=1')
        self.documents[1]['invalidated_at']=100
        ev.mutate(self.db,self.documents)

    def test_seeds_change_order_without_changing_documents(self):
        first=ev.corpus(1000,0); second=ev.corpus(1000,1)
        self.assertEqual(first,second)
        self.assertNotEqual(list(first),list(second))

    def test_explicit_boundary_judgments(self):
        queries={q['name']:q for q in ev.FIXTURE['queries']}
        self.assertEqual(ev.expected(self.documents,queries['deploy']),{1,2,3,4})
        self.assertEqual(ev.expected(self.documents,queries['before-start']),{2,3,4})
        self.assertEqual(ev.expected(self.documents,queries['at-expiry']),{2,3,4})
        self.assertEqual(ev.expected(self.documents,queries['absent-tenant']),set())
        self.assert_queries()

    def test_pending_writes_savepoint_and_full_rollback(self):
        before=self.answers(); original=copy.deepcopy(self.documents)
        self.db.execute('savepoint trial');self.change();self.assert_queries()
        self.db.execute('rollback to trial');self.db.execute('release trial')
        self.documents=copy.deepcopy(original);self.assert_queries()
        self.assertEqual(self.answers(),before)
        self.db.execute('begin');self.change();self.assert_queries()
        self.db.execute('rollback');self.documents=original;self.assert_queries()
        self.assertEqual(self.answers(),before)
        self.db.close();self.db=ev.connect(self.path,EXTENSION)
        self.assertEqual(self.answers(),before);self.assert_contents()

    def test_commit_and_reopen(self):
        self.db.execute('begin');self.change();self.assert_queries();self.db.execute('commit')
        answers=self.answers();self.db.close();self.db=ev.connect(self.path,EXTENSION)
        self.assert_queries();self.assertEqual(self.answers(),answers)
        self.assertEqual(ev.expected(self.documents,ev.FIXTURE['queries'][0]),{2,3})

    def test_unsafe_filter_mutations_are_detected(self):
        query=ev.FIXTURE['queries'][0]
        mutations=[('tenant = :tenant','1=1'),('invalidated_at is null','1=1'),('valid_to > :as_of','valid_to >= :as_of'),('valid_from <= :as_of','1=1')]
        for old,new in mutations:
            with self.subTest(mutation=old):
                self.assertEqual(ev.HYBRID.count(old),1)
                rows=self.db.execute(ev.HYBRID.replace(old,new),ev.params(query,10)).fetchall()
                self.assertTrue(ev.assess(rows,self.documents,query)['ineligible_ids'])

    def test_late_filter_control_loses_relevant_candidates(self):
        query=ev.FIXTURE['queries'][0]
        late=self.db.execute(ev.LATE,ev.params(query,10)).fetchall()
        safe=ev.retrieve(self.db,'hybrid',query,10)
        self.assertLess(len({r[0] for r in late}&ev.expected(self.documents,query)),
                        len({r[0] for r in safe}&ev.expected(self.documents,query)))


if __name__=='__main__':
    unittest.main()
