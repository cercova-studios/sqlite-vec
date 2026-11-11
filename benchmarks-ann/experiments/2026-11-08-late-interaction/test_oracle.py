"""L0 scalar contract tests; request-wide ranking/publication belongs to the runner."""

from contextlib import closing
from dataclasses import FrozenInstanceError
import math
import random
import sqlite3
import struct
import tracemalloc
import unittest

import oracle


def pack(rows):
    return b"".join(struct.pack("<f", value) for row in rows for value in row)


def unpack(blob, dimensions):
    values = [value for (value,) in struct.iter_unpack("<f", blob)]
    return [values[i:i + dimensions] for i in range(0, len(values), dimensions)]


def relational_score(query, document):
    """Independent ordinary-SQL SUM, then MAX, then SUM; no vec extension."""
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("CREATE TABLE q(token INTEGER,component INTEGER,value REAL)")
        db.execute("CREATE TABLE d(token INTEGER,component INTEGER,value REAL)")
        for table, rows in [("q", query), ("d", document)]:
            db.executemany(f"INSERT INTO {table} VALUES(?,?,?)",
                           [(token, component, value) for token, row in enumerate(rows)
                            for component, value in enumerate(row)])
        return db.execute("""
            WITH dots AS (
              SELECT q.token AS qid,d.token AS did,SUM(q.value*d.value) AS dot
              FROM q JOIN d USING(component) GROUP BY q.token,d.token
            ), maxima AS (SELECT qid,MAX(dot) AS best FROM dots GROUP BY qid)
            SELECT SUM(best) FROM maxima
        """).fetchone()[0]


class OracleTests(unittest.TestCase):
    def score(self, query, document):
        return oracle.maxsim(pack(query), pack(document), len(query[0]), oracle.MAX_WORK).score

    def test_hand_scores_negative_matches_and_metadata(self):
        query = pack([(1, 0), (0, 1)])
        document = pack([(1, 0), (0, 1), (-1, 0)])
        result = oracle.maxsim(query, document, 2, 30)
        self.assertEqual(result, oracle.Score(2.0, 30, 2, 3, 2))
        self.assertEqual(oracle.work_required(query, document, 2), 30)
        with self.assertRaises(FrozenInstanceError):
            result.score = 0
        self.assertEqual(self.score([(1, 0)], [(0, 1)]), 0)
        self.assertEqual(self.score([(1, 0)], [(-1, 0)]), -1)
        negative = unpack(pack([(-0.6, -0.8)]), 2)[0][0]
        self.assertEqual(self.score([(1, 0)], [(-1, 0), (-0.6, -0.8)]), negative)

    def test_query_multiplicity_document_duplicates_and_permutation(self):
        query = [(1, 0), (1, 0), (0, 1)]
        document = [(1, 0), (0, 1), (-1, 0)]
        self.assertEqual(self.score(query, document), 3)
        self.assertEqual(self.score(query, document[::-1]), 3)
        self.assertEqual(self.score(query, document + document), 3)
        self.assertEqual(self.score(query + query, document), 6)

    def test_little_endian_is_explicit(self):
        one = b"\x00\x00\x80\x3f"
        minus_one = b"\x00\x00\x80\xbf"
        self.assertEqual(oracle.maxsim(one, minus_one, 1, 5).score, -1)
        with self.assertRaisesRegex(ValueError, "norm"):
            oracle.maxsim(struct.pack(">f", 1), one, 1, 5)

    def test_float64_order_and_no_clamping(self):
        row = [1.0, 2**-13, 2**-13]
        expected = 1.0 + 2**-25
        self.assertEqual(self.score([row], [row]), expected)
        self.assertGreater(expected, 1)
        self.assertEqual(self.score([row] * 128, [row]), 128 * expected)
        # Deliberately round every addition to float32: the oracle must reject parity.
        bad = 0.0
        for value in row:
            bad = struct.unpack("<f", struct.pack("<f", bad + value * value))[0]
        with self.assertRaises(AssertionError):
            self.assertEqual(self.score([row], [row]), bad)
        tiny = 2**-60
        query = [(1, 0), (tiny, 1), (-1, 0)]
        self.assertEqual(self.score(query, [(1, 0)]), 0)
        self.assertEqual(self.score([query[0], query[2], query[1]], [(1, 0)]), tiny)
        a = struct.unpack("<f", struct.pack("<f", math.sqrt(0.5)))[0]
        self.assertEqual(self.score([(a, 2**-30, a)], [(a, 2**-30, -a)]), 0)
        self.assertEqual(self.score([(a, a, 2**-30)], [(a, -a, 2**-30)]), tiny)

    def test_invalid_types_shapes_and_huge_integer_bounds(self):
        one = pack([(1,)])
        for dimension in [True, False, 0, -1, 4097, 1 << 128, 1.0, "1", None]:
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                oracle.work_required(one, one, dimension)
        for value in [None, True, "bytes", [], bytearray(one), memoryview(one), b"", b"\0", one + b"\0"]:
            for query, document in [(value, one), (one, value)]:
                with self.subTest(value=type(value)), self.assertRaises(ValueError):
                    oracle.maxsim(query, document, 1, oracle.MAX_WORK)
        for limit in [True, False, -1, oracle.MAX_WORK + 1, 1 << 128, 8.0, "8", None]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                oracle.maxsim(one, one, 1, limit)
        with self.assertRaisesRegex(ValueError, "complete rows"):
            oracle.maxsim(one, one, 2, oracle.MAX_WORK)

    def test_every_row_is_checked_for_finite_components_and_norms(self):
        valid = [(1.0, 0.0)]
        for invalid, message in [([0, -0.0], "norm"), ([1, 1], "norm"),
                                 ([3.4e38, 0], "norm"), ([math.nan, 0], "nonfinite"),
                                 ([math.inf, 0], "nonfinite"), ([-math.inf, 0], "nonfinite")]:
            bad = valid + [invalid]
            for query, document in [(bad, valid), (valid, bad)]:
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, message):
                    oracle.maxsim(pack(query), pack(document), 2, oracle.MAX_WORK)

    def test_adjacent_float32_norm_boundary(self):
        below = struct.unpack("<f", struct.pack("<f", 0.01))[0]
        bits = struct.unpack("<I", struct.pack("<f", below))[0]
        above = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
        self.assertLess(abs((1 + below * below) - 1), 1e-4)
        self.assertGreater(abs((1 + above * above) - 1), 1e-4)
        oracle.maxsim(pack([(1, below)]), pack([(1, 0)]), 2, oracle.MAX_WORK)
        with self.assertRaisesRegex(ValueError, "norm"):
            oracle.maxsim(pack([(1, above)]), pack([(1, 0)]), 2, oracle.MAX_WORK)

    def test_rows_dimensions_and_byte_ceiling_boundaries(self):
        one = pack([(1,)])
        self.assertEqual(oracle.maxsim(one * 128, one, 1, oracle.MAX_WORK).query_rows, 128)
        self.assertEqual(oracle.maxsim(one, one * 4096, 1, oracle.MAX_WORK).document_rows, 4096)
        for q, d, role in [(one * 129, one, "query"), (one, one * 4097, "document")]:
            with self.assertRaisesRegex(ValueError, role + " rows"):
                oracle.work_required(q, d, 1)
        row = one + bytes(4095 * 4)
        self.assertEqual(oracle.maxsim(row, row, 4096, oracle.MAX_WORK).score, 1)
        query = bytes(oracle.MAX_QUERY_BYTES)
        document = bytes(oracle.MAX_DOCUMENT_BYTES)
        self.assertGreater(oracle.work_required(query, bytes(2048 * 4), 2048), 0)
        self.assertGreater(oracle.work_required(bytes(512 * 4), document, 512), 0)
        for q, d, dim, role in [(query + b"\0", bytes(2048 * 4), 2048, "query"),
                                 (bytes(512 * 4), document + b"\0", 512, "document")]:
            with self.assertRaisesRegex(ValueError, role + " bytes"):
                oracle.work_required(q, d, dim)

    def test_work_admission_precedes_decode_and_has_one_unit_boundary(self):
        query = pack([(1, 0), (0, 1)])
        document = pack([(1, 0), (0, 1), (-1, 0)])
        self.assertEqual(oracle.maxsim(query, document, 2, 30).work_used, 30)
        for limit in [0, 29]:
            with self.assertRaises(oracle.WorkLimitError):
                oracle.maxsim(query, document, 2, limit)
        zeros = bytes(8)
        self.assertEqual(oracle.work_required(zeros, zeros, 2), 8)
        with self.assertRaises(oracle.WorkLimitError):
            oracle.maxsim(zeros, zeros, 2, 7)
        with self.assertRaisesRegex(ValueError, "norm"):
            oracle.maxsim(zeros, zeros, 2, 8)
        query = bytes(128 * 512 * 4)
        document = bytes(4096 * 512 * 4)
        self.assertGreater(oracle.work_required(query, document, 512), oracle.MAX_WORK)
        with self.assertRaises(oracle.WorkLimitError):
            oracle.maxsim(query, document, 512, oracle.MAX_WORK)
        # A rejected call cannot poison a later call or return a partial Score.
        self.assertEqual(oracle.maxsim(pack([(1,)]), pack([(1,)]), 1, 5).score, 1)

    def test_relational_oracle_on_seeded_normalized_float32_pairs(self):
        rng = random.Random(20261108)
        for _ in range(20):
            dimensions, m, n = rng.randint(1, 6), rng.randint(1, 4), rng.randint(1, 5)
            rows = []
            for _ in range(m + n):
                row = [rng.uniform(-1, 1) for _ in range(dimensions)]
                norm = math.sqrt(sum(value * value for value in row))
                rows.append([value / norm for value in row])
            q, d = pack(rows[:m]), pack(rows[m:])
            expected = relational_score(unpack(q, dimensions), unpack(d, dimensions))
            self.assertAlmostEqual(oracle.maxsim(q, d, dimensions, oracle.MAX_WORK).score,
                                   expected, places=12)

    def test_deliberately_wrong_maximum_and_omission_controls_fail(self):
        def wrong_zero(query, document):
            return sum(max(0.0, max(sum(a * b for a, b in zip(q, d)) for d in document))
                       for q in query)
        cases = [
            ([(1, 0)], [(-1, 0)], wrong_zero),
            ([(1, 0), (0, 1)], [(1, 0), (0, 1)], lambda q, d: relational_score(q[:-1], d)),
            ([(1, 0)], [(-1, 0), (1, 0)], lambda q, d: relational_score(q, d[:-1])),
            ([(1, 0), (1, 0)], [(1, 0)], lambda q, d: relational_score(list(dict.fromkeys(q)), d)),
        ]
        for query, document, wrong in cases:
            actual, corrupted = self.score(query, document), wrong(query, document)
            with self.subTest(query=query), self.assertRaises(AssertionError):
                self.assertEqual(actual, corrupted)

    def test_scratch_does_not_scale_with_document_components(self):
        query = pack([(1, 0, 0, 0)])
        document = query * 4096
        already_tracing = tracemalloc.is_tracing()
        if not already_tracing:
            tracemalloc.start()
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        try:
            result = oracle.maxsim(query, document, 4, oracle.MAX_WORK)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if not already_tracing:
                tracemalloc.stop()
        self.assertEqual(result.score, 1)
        self.assertLess(peak - baseline, 64 * 1024)


if __name__ == "__main__":
    unittest.main()
