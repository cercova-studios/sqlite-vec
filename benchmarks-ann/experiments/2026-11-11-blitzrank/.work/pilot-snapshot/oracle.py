"""Strict, logged RankGPT comparisons for the pinned upstream BlitzRank sorter."""
import json
import os
from pathlib import Path
import re
import time
import urllib.request


class Budget:
    """Durable reservations include unknown charges from interrupted requests."""
    def __init__(self, path, cap_nusd=5_000_000_000):
        self.path = Path(path)
        self.cap = cap_nusd
        self.state = json.loads(self.path.read_text()) if self.path.exists() else dict(cap_nusd=cap_nusd, entries=[])
        if self.state["cap_nusd"] != cap_nusd:
            raise ValueError("budget cap changed")

    def persist(self):
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(self.state, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)

    def reserve(self, payload):
        # One token per UTF-8 byte plus 4096 framing tokens overestimates this text-only request.
        maximum_input = len(json.dumps(payload, ensure_ascii=False).encode()) + 4096
        if maximum_input > 272000:
            raise ValueError("request exceeds the standard-price input bound")
        amount = maximum_input * 125 + payload["max_completion_tokens"] * 500
        if sum(e["charged_nusd"] for e in self.state["entries"]) + amount > self.cap:
            raise RuntimeError("$5 budget cannot cover the next request's worst-case charge")
        self.state["entries"].append(dict(charged_nusd=amount, reserved_nusd=amount,
                                         maximum_input_tokens=maximum_input, settled=False))
        self.persist()
        return len(self.state["entries"]) - 1

    def settle(self, index, response):
        tokens = usage(response)
        amount = tokens["input_tokens"] * 125 + tokens["output_tokens"] * 500
        entry = self.state["entries"][index]
        if amount > entry["reserved_nusd"]:
            raise RuntimeError("provider token usage exceeded the reserved cost bound")
        entry.update(charged_nusd=amount, settled=True, usage=tokens)
        self.persist()


def permutation(text, count):
    if not isinstance(text, str) or not re.fullmatch(r"\s*\[\d+\](?:\s*>\s*\[\d+\])*\s*", text):
        raise ValueError("model response is not a bracketed ranking")
    indices = [int(n) - 1 for n in re.findall(r"\[(\d+)\]", text)]
    if len(indices) != count or set(indices) != set(range(count)):
        raise ValueError("model ranking has missing, duplicate or out-of-range identifiers")
    return indices


def messages(query, docs):
    # RankGPT prompt from the pinned upstream listwise_rank_gpt.py.
    result = [
        {"role": "system", "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
        {"role": "user", "content": f"I will provide you with {len(docs)} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
        {"role": "assistant", "content": "Okay, please provide the passages."},
    ]
    for rank, doc in enumerate(docs, 1):
        result.extend([
            {"role": "user", "content": f"[{rank}] {doc}"},
            {"role": "assistant", "content": f"Received passage [{rank}]."},
        ])
    result.append({"role": "user", "content": f"Search Query: {query}. \nRank the {len(docs)} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."})
    return result


def usage(response):
    value = response["usage"]
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError("response must report nonnegative integer token usage")
    if value["total_tokens"] != value["prompt_tokens"] + value["completion_tokens"]:
        raise ValueError("inconsistent token usage")
    if value["prompt_tokens"] == 0 or value["completion_tokens"] == 0:
        raise ValueError("model comparison must consume input and output tokens")
    reasoning = value.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    if type(reasoning) is not int or not 0 <= reasoning <= value["completion_tokens"]:
        raise ValueError("invalid reasoning token subset")
    return dict(input_tokens=value["prompt_tokens"], output_tokens=value["completion_tokens"],
                total_tokens=value["total_tokens"], reasoning_tokens=reasoning)


def compare(protocol, query, docs, log, post=None, budget=None):
    payload = dict(model=protocol["model"], messages=messages(query, docs),
                   reasoning_effort=protocol.get("reasoning_effort", "low"),
                   max_completion_tokens=protocol["max_output_tokens"], store=False, service_tier="default")
    reservation = budget.reserve(payload) if budget is not None else None
    start = time.perf_counter_ns()
    if post is None:
        if protocol["endpoint"] != "https://api.openai.com/v1" or budget is None:
            raise ValueError("live calls require the OpenAI endpoint and a durable budget")
        request = urllib.request.Request(protocol["endpoint"] + "/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
        with urllib.request.urlopen(request, timeout=protocol["call_timeout_seconds"]) as response:
            result = json.load(response)
    else:
        result = post(payload)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    # Persist even malformed replies before validation; failures still consume tokens.
    log(dict(request=payload, response=result, latency_ms=elapsed, budget_entry=reservation))
    if budget is not None:
        budget.settle(reservation, result)
    tokens = usage(result)
    if result.get("model") != protocol["model"]:
        raise ValueError("server returned a different model identity")
    choice = result["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("model generation did not finish normally")
    return permutation(choice["message"]["content"], len(docs)), tokens
