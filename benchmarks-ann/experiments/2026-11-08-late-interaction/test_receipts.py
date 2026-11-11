"""Preserved benchmark receipts must reject incomplete or falsified results."""
import copy
import gzip
import json
from pathlib import Path
import unittest

from analyze import verify_rows


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent / "recorded"
        cls.score = json.loads(gzip.decompress((root / "score.json.gz").read_bytes()))
        cls.live = json.loads(gzip.decompress((root / "live.json.gz").read_bytes()))
        cls.protocol = json.loads((root / "protocol.json").read_text())
        cls.inventory = json.loads((root / "inventory.json").read_text())

    def test_complete_receipts_and_real_embedding_oracles(self):
        self.assertEqual(len(verify_rows(self.score, self.live, self.protocol, self.inventory)), 600)

    def test_wrong_rank_metric_work_and_missing_timing_are_detected(self):
        for corruption in ["rank", "metric", "work", "timing"]:
            with self.subTest(corruption=corruption):
                score, live = copy.deepcopy(self.score), copy.deepcopy(self.live)
                if corruption == "rank":
                    score["records"][0]["scoring"]["scores"].reverse()
                elif corruption == "metric":
                    score["records"][0]["late"]["ndcg10"] += .1
                elif corruption == "work":
                    score["records"][0]["scoring"]["work_used"] -= 1
                else:
                    live["timings"].pop()
                with self.assertRaises(AssertionError):
                    verify_rows(score, live, self.protocol, self.inventory)


if __name__ == "__main__":
    unittest.main()
