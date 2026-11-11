"""Failure, accounting and upstream-algorithm checks without model inference."""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import unittest
import threading

import experiment
from oracle import Budget, compare, messages, permutation, usage
from verify import relevance


def response(text="[2] > [1]", finish="stop"):
    return dict(model="fixture", choices=[dict(finish_reason=finish, message=dict(content=text))],
                usage=dict(prompt_tokens=100, completion_tokens=20, total_tokens=120,
                           completion_tokens_details=dict(reasoning_tokens=5)))


class Contracts(unittest.TestCase):
    def test_parallel_workers_cannot_double_spend_the_remaining_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = Budget(Path(directory)/"budget.json",2_000_000)
            barrier = threading.Barrier(2)
            def reserve():
                barrier.wait()
                try:
                    return budget.reserve(dict(messages=[],max_completion_tokens=2048))
                except RuntimeError:
                    return None
            with ThreadPoolExecutor(max_workers=2) as pool:
                result = list(pool.map(lambda _:reserve(),range(2)))
            self.assertEqual(sum(r is not None for r in result),1)
            self.assertEqual(len(budget.state["entries"]),1)
            self.assertLessEqual(sum(e["charged_nusd"] for e in budget.state["entries"]),budget.cap)

    def test_reuse_refuses_changed_inference_settings(self):
        previous = {k:1 for k in experiment.REUSE_FIELDS}
        experiment.check_reuse_settings(previous,dict(previous))
        for name in ["model","window_size","max_document_tokens","api_cap_usd"]:
            with self.subTest(name=name),self.assertRaises(ValueError):
                experiment.check_reuse_settings(previous,dict(previous,**{name:2}))

    def test_independent_rank_position_metrics(self):
        measured = relevance([1,2,3],{"2":1,"4":1},[1,2,3,4])
        self.assertAlmostEqual(measured["ndcg10"],(1/1.584962500721156)/(1+1/1.584962500721156))
        self.assertEqual(measured["mrr10"],.5)
        self.assertEqual(measured["recall10"],.5)
        self.assertEqual(measured["candidate_recall"],1)

    def test_budget_reserves_before_inference_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"budget.json"
            budget = Budget(path, 2_000_000)
            payload = dict(messages=[],max_completion_tokens=2048)
            index = budget.reserve(payload)
            self.assertFalse(Budget(path,2_000_000).state["entries"][0]["settled"])
            with self.assertRaises(RuntimeError):
                budget.reserve(payload)
            budget.settle(index,response())
            self.assertEqual(Budget(path,2_000_000).state["entries"][0]["charged_nusd"],22500)
            protocol = dict(model="fixture",endpoint="unused",max_output_tokens=2048,call_timeout_seconds=1)
            posts = []
            with self.assertRaises(RuntimeError):
                compare(protocol,"query",["one","two"],lambda _:None,
                        post=lambda p:posts.append(p),budget=Budget(Path(directory)/"small.json",1))
            self.assertEqual(posts,[])

    def test_rankings_are_complete_permutations(self):
        self.assertEqual(permutation(" [2] > [1]\n", 2), [1, 0])
        for text in ["[1]", "[1] > [1]", "[0] > [2]", "[1] > [3]", "Here: [2] > [1]", "", "[1.2]"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                permutation(text, 2)

    def test_reasoning_is_a_subset_not_an_extra_charge(self):
        self.assertEqual(usage(response()), dict(input_tokens=100, output_tokens=20,
                         total_tokens=120, reasoning_tokens=5))
        for update in [dict(total_tokens=125), dict(prompt_tokens=-1), dict(completion_tokens=True),
                       dict(completion_tokens_details=dict(reasoning_tokens=21))]:
            bad = response()
            bad["usage"].update(update)
            with self.assertRaises(ValueError):
                usage(bad)

    def test_failed_generation_is_logged_and_never_repaired(self):
        protocol = dict(model="fixture", endpoint="unused", max_output_tokens=100,
                        call_timeout_seconds=1)
        for bad in [response("[1]"), response(finish="length"), dict(response(), model="different")]:
            calls = []
            with self.assertRaises(ValueError):
                compare(protocol, "query", ["one", "two"], calls.append, post=lambda _:bad)
            self.assertEqual(len(calls), 1)
            self.assertEqual(usage(calls[0]["response"])["total_tokens"], 120)
        calls = []
        indices, tokens = compare(protocol,"query",["one","two"],calls.append,post=lambda _:response())
        self.assertEqual(indices,[1,0])
        self.assertEqual(tokens["total_tokens"],120)

    def test_prompt_matches_pinned_upstream(self):
        source = experiment.HERE/".work/upstream/blitzrank/engine/comparer/listwise_rank_gpt.py"
        tree = ast.parse(source.read_text())
        names = {"get_prefix_prompt","get_post_prompt","create_permutation_instruction"}
        functions = ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
        namespace = {}
        exec("from typing import List, Dict", namespace)
        exec(compile(functions,str(source),"exec"),namespace)
        self.assertEqual(messages("Q",["first","second"]),
                         namespace["create_permutation_instruction"]("Q",["first","second"]))

    def test_official_sorter_uses_comparisons_and_finds_top_three(self):
        sys.path.insert(0,str(experiment.HERE/".work/upstream/blitzrank/engine/algorithms"))
        from tournament_graph.compare_oracle import CompareOracle, Item, OracleResult, make_edges_from_linear_order
        from tournament_graph.tournament_graph_sort import tournament_graph_sort_async

        class KnownOrder(CompareOracle):
            async def _compare_k_async(self, items):
                return OracleResult(make_edges_from_linear_order(sorted(items,key=lambda i:i.id)))

        oracle = KnownOrder(5)
        result = asyncio.run(tournament_graph_sort_async([Item(i) for i in range(24,-1,-1)],oracle,3))
        self.assertEqual([i.id for i in result.results],[0,1,2])
        self.assertGreater(oracle.calls,1)
        self.assertEqual(result.num_oracle_calls,oracle.calls)

    def test_incomplete_and_tampered_receipts_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = dict(qrels={"q":{"2":1}})
            _,_,metric = experiment.prior_modules()
            rows = [dict(query="q",method=m,candidates=[1,2],ids=[1,2],
                         metrics=metric.metrics([1,2],{"2":1},[1,2])) for m in experiment.METHODS]
            experiment.write(root/"inputs.json.gz",inputs)
            experiment.write(root/"baselines.json.gz",dict(rows=rows,timings=[dict(method=m,total_ms=1) for m in experiment.METHODS]))
            protocol = dict(inputs_sha256=experiment.sha(root/"inputs.json.gz"),
                            baselines_sha256=experiment.sha(root/"baselines.json.gz"),scripts={},
                            evaluation_ids=["q"],split_caveat="fixture",selected_baseline="dense")
            experiment.write(root/"protocol.json",protocol)
            (root/"results").mkdir()
            self.assertFalse(experiment.summary(root)["complete"])
            for m in experiment.METHODS:
                experiment.write(root/"results"/(m+".json.gz"),
                    dict(query="q",method=m,status="ok",protocol_sha256=experiment.sha(root/"protocol.json"),
                         candidates=[1,2],ids=[2,1],calls=[],total_ms=2))
            # Empty LLM logs must not count as evidence that reranking ran.
            with self.assertRaises(ValueError):
                experiment.summary(root)


if __name__ == "__main__":
    unittest.main()
