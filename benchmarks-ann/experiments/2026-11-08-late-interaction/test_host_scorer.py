"""Independent host-backend, request-budget, publication and metric checks."""
import hashlib
import math
import random
import sqlite3
import struct
import unittest
from unittest.mock import patch

import numpy as np

import host_scorer as host
from metrics import bootstrap, metrics
from oracle import maxsim, work_required


def pack(rows):
    return b"".join(struct.pack("<f", v) for row in rows for v in row)


class HostTests(unittest.TestCase):
    def pair(self, q, d):
        qb, db, dimensions = pack(q), pack(d), len(q[0])
        actual = host.score_pair(host.decode(qb,dimensions,128,1048576),host.decode(db,dimensions,4096,8388608))
        expected = maxsim(qb,db,dimensions,134217728)
        self.assertEqual(host.pair_work(len(q),len(d),dimensions),expected.work_used)
        self.assertEqual(expected.work_used,work_required(qb,db,dimensions))
        self.assertEqual(actual,expected.score)

    def test_ordered_float64_matches_scalar_with_cancellation_and_negative_scores(self):
        self.pair([(1,0)], [(-1,0)])
        self.pair([(1,0),(2**-60,1),(-1,0)], [(1,0)])
        self.pair([(1,0),(-1,0),(2**-60,1)], [(1,0)])
        a=struct.unpack("<f",struct.pack("<f",math.sqrt(.5)))[0]
        self.pair([(a,2**-30,a)], [(a,2**-30,-a)])
        self.pair([(a,a,2**-30)], [(a,-a,2**-30)])
        self.pair([(1,2**-13,2**-13)],[(1,2**-13,2**-13)])

    def test_seeded_pairs_and_permutation_match_scalar_exactly(self):
        rng=random.Random(20261108)
        for _ in range(60):
            dim,m,n=rng.randint(1,8),rng.randint(1,5),rng.randint(1,6)
            rows=[]
            for _ in range(m+n):
                row=[rng.uniform(-1,1) for _ in range(dim)]
                length=math.sqrt(sum(v*v for v in row))
                rows.append([v/length for v in row])
            self.pair(rows[:m],rows[m:])
            self.pair(rows[:m],list(reversed(rows[m:])))

    def fixture(self):
        db=sqlite3.connect(":memory:",isolation_level=None)
        self.addCleanup(db.close)
        db.executescript("""CREATE TABLE sources(id INTEGER PRIMARY KEY,body TEXT,source_revision INTEGER);
          CREATE TABLE li_artifacts(artifact_id INTEGER PRIMARY KEY,sha256 TEXT,profile_id TEXT,role TEXT,
           token_count INTEGER,dimensions INTEGER,input_sha256 TEXT,vectors BLOB);
          CREATE TABLE li_refs(role TEXT,item_id INTEGER,artifact_sha256 TEXT,source_revision INTEGER);""")
        profile="0"*64
        def insert(role,key,text,rows):
            blob=pack(rows);source=hashlib.sha256(text.encode()).hexdigest()
            sha=host.digest_artifact(profile,role,source,len(rows),2,blob)
            db.execute("INSERT INTO li_artifacts(sha256,profile_id,role,token_count,dimensions,input_sha256,vectors) VALUES(?,?,?,?,?,?,?)",
                       (sha,profile,role,len(rows),2,source,blob))
            db.execute("INSERT INTO li_refs VALUES(?,?,?,1)",(role,key,sha))
            if role=="document":db.execute("INSERT INTO sources VALUES(?,?,1)",(key,text))
        insert("query",42,"query",[(1,0)])
        insert("document",-2,"negative",[(-1,0)])
        insert("document",0,"orthogonal",[(0,1)])
        insert("document",7,"positive",[(1,0)])
        db.execute("BEGIN")
        return db,profile

    def test_request_ranks_negative_scores_and_counts_real_work(self):
        db,p=self.fixture()
        result=host.rerank(db,42,"query",[-2,0,7],p)
        self.assertEqual(result["scores"],[(7,1.0),(0,0.0),(-2,-1.0)])
        self.assertEqual(result["scored_candidates"],3)
        self.assertEqual(result["component_products"],6)
        self.assertEqual(result["work_used"],2+3*8)
        self.assertEqual(result["candidate_bytes"],24)
        self.assertEqual(result["matrix_bytes_read"],32)
        self.assertNotEqual([r[0] for r in result["scores"]],[-2,0,7])

    def test_aggregate_limit_and_reset_per_document_negative_control(self):
        db,p=self.fixture()
        with self.assertRaisesRegex(ValueError,"aggregate"):
            host.rerank(db,42,"query",[-2,0,7],p,work_limit=25)
        def wrong_reset(m,d,metadata,limit):
            return max(host.pair_work(m,r["tokens"],d) for r in metadata),sum(r["bytes"] for r in metadata)
        with patch.object(host,"request_work",wrong_reset),self.assertRaises(AssertionError):
            with self.assertRaises(ValueError):
                host.rerank(db,42,"query",[-2,0,7],p,work_limit=25)
        self.assertEqual(host.rerank(db,42,"query",[-2,0,7],p,work_limit=26)["work_used"],26)

    def test_missing_stale_model_digest_duplicate_and_no_partial_publication(self):
        for mutation,fragment in [("DELETE FROM li_refs WHERE role='document' AND item_id=7","missing"),
                                  ("UPDATE sources SET source_revision=2 WHERE id=7","epoch"),
                                  ("UPDATE li_artifacts SET profile_id='wrong' WHERE role='document'","profile"),
                                  ("UPDATE li_artifacts SET vectors=zeroblob(8) WHERE role='document'","digest")]:
            with self.subTest(mutation=mutation):
                db,p=self.fixture();db.execute(mutation)
                with self.assertRaisesRegex(ValueError,fragment):host.rerank(db,42,"query",[-2,0,7],p)
        db,p=self.fixture()
        with self.assertRaises(ValueError):host.rerank(db,42,"query",[7,7],p)
        calls=[]
        def second_fails(q,d):
            calls.append(1)
            if len(calls)==2:raise ValueError("second candidate failure")
            return 1.0
        with patch.object(host,"score_pair",second_fails),self.assertRaisesRegex(ValueError,"second candidate"):
            host.rerank(db,42,"query",[-2,0,7],p)
        self.assertEqual(len(calls),2)

    def test_invalid_shapes_norms_and_global_caps(self):
        one=pack([(1,0)])
        for blob,dim in [(b"",2),(one,True),(one,0),(one+b"x",2),(pack([(0,0)]),2),(pack([(float('nan'),0)]),2)]:
            with self.assertRaises(ValueError):host.decode(blob,dim,128,1048576)
        metadata=[dict(id=1,tokens=1,bytes=8)]
        for limit in [0,True,-1,134217729]:
            with self.assertRaises(ValueError):host.request_work(1,2,metadata,limit)
        with self.assertRaises(ValueError):host.request_work(1,2,metadata*2,100)
        with self.assertRaises(ValueError):host.request_work(1,2,[dict(id=True,tokens=1,bytes=8)],100)

    def test_request_admission_precedes_payload_hashing_and_numeric_decode(self):
        db,p=self.fixture()
        with patch.object(host,"decode",side_effect=AssertionError("decoded before admission")):
            with self.assertRaisesRegex(ValueError,"aggregate"):
                host.rerank(db,42,"query",[-2,0,7],p,work_limit=1)
        db.execute("UPDATE li_artifacts SET vectors=zeroblob(1048584) WHERE role='query'")
        with patch.object(host,"digest_artifact",side_effect=AssertionError("hashed oversized query")):
            with self.assertRaisesRegex(ValueError,"query shape"):
                host.rerank(db,42,"query",[-2,0,7],p)
        for ids in [[True],[7]*101,[2**63]]:
            with patch.object(host,"digest_artifact",side_effect=AssertionError("hashed before ID admission")):
                with self.assertRaises(ValueError):host.rerank(db,42,"query",ids,p)

    def test_empty_candidates_still_admit_and_validate_query(self):
        db,p=self.fixture()
        result=host.rerank(db,42,"query",[],p,work_limit=2)
        self.assertEqual(result["scores"],[])
        self.assertEqual(result["scored_candidates"],0)
        self.assertEqual(result["work_used"],2)
        self.assertEqual(result["matrix_bytes_read"],8)
        with self.assertRaises(ValueError):host.rerank(db,42,"query",[],p,work_limit=1)
        blob=pack([(0,0)])
        sha=db.execute("SELECT sha256 FROM li_artifacts WHERE role='query'").fetchone()[0]
        source=hashlib.sha256(b"query").hexdigest()
        changed=host.digest_artifact(p,"query",source,1,2,blob)
        db.execute("UPDATE li_artifacts SET vectors=?,sha256=? WHERE role='query'",(blob,changed))
        db.execute("UPDATE li_refs SET artifact_sha256=? WHERE artifact_sha256=?",(changed,sha))
        with self.assertRaisesRegex(ValueError,"unnormalized"):
            host.rerank(db,42,"query",[],p)

    def test_metrics_have_hand_computed_oracles_and_bootstrap_controls(self):
        result=metrics([3,1,2],{"1":1,"2":1},[3,1,2])
        self.assertEqual(result["mrr10"],.5)
        self.assertEqual(result["recall10"],1)
        self.assertAlmostEqual(result["ndcg10"],(.5+1/math.log2(3))/(1+1/math.log2(3)))
        self.assertEqual(metrics([3],{"1":1},[3])["ndcg10"],0)
        self.assertEqual(bootstrap([.25]*5,samples=100),dict(mean=.25,low=.25,high=.25,samples=100,seed=20261110))
        self.assertEqual(bootstrap([0]*5,samples=100)["low"],0)
        with self.assertRaises(ValueError):metrics([1,1],{"1":1})


if __name__=="__main__":unittest.main()
