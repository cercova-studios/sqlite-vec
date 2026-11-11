"""Audit real model-call receipts, relevance arithmetic and budget accounting."""
import json
import math
import statistics

from experiment import HERE, METHODS, read, sha, summary, verify_inputs
from oracle import messages, permutation, usage


def relevance(ids, judgments, candidates):
    positives = {int(key) for key, grade in judgments.items() if grade > 0}
    if not positives or len(ids) != len(set(ids)) or len(ids) > 10:
        raise ValueError("invalid relevance inputs")
    hit_positions = [rank for rank, key in enumerate(ids,1) if key in positives]
    ideal = sum(1/math.log2(rank+1) for rank in range(1,min(10,len(positives))+1))
    return dict(ndcg10=sum(1/math.log2(rank+1) for rank in hit_positions)/ideal,
                mrr10=1/hit_positions[0] if hit_positions else 0,
                recall10=len(hit_positions)/len(positives),
                candidate_recall=len(set(candidates)&positives)/len(positives))


def main():
    recorded = HERE/"recorded"
    protocol, inputs, baseline = verify_inputs(recorded)
    report = summary(recorded)
    if not report["complete"]:
        raise ValueError("experiment is incomplete or contains failed requests")
    if list((recorded/"results").glob("*.partial")):
        raise ValueError("interrupted response journals remain")
    ledger = read(recorded/"budget.json")
    if ledger["cap_nusd"] != 5_000_000_000:
        raise ValueError("user spending cap changed")
    records = [read(p) for p in (recorded/"results").glob("*.json.gz")]
    indices = []
    import tiktoken
    tokenizer = tiktoken.get_encoding(protocol["tokenizer"])
    for row in baseline["rows"]:
        independent = relevance(row["ids"],inputs["qrels"][row["query"]],row["candidates"])
        if any(abs(independent[k]-row["metrics"][k]) > 1e-14 for k in independent):
            raise ValueError("baseline metrics disagree with independent rank-position oracle")
    for row in records:
        if len(row["ids"]) != min(protocol["topk"],len(row["candidates"])):
            raise ValueError("incorrect final ranking length")
        if row["total_ms"] < sum(call["latency_ms"] for call in row["calls"]):
            raise ValueError("request latency excludes model calls")
        query = tokenizer.decode(tokenizer.encode(inputs["queries"][row["query"]].strip(),disallowed_special=())[:1024])
        for call, round_log in zip(row["calls"],row["rounds"]):
            index = call["budget_entry"]
            indices.append(index)
            entry = ledger["entries"][index]
            tokens = usage(call["response"])
            cached = call["response"]["usage"].get("prompt_tokens_details",{}).get("cached_tokens",0)
            if type(cached) is not int or not 0 <= cached <= tokens["input_tokens"]:
                raise ValueError("cached token count is not a valid input subset")
            if not entry["settled"] or entry["usage"] != tokens:
                raise ValueError("model-call usage does not match settled budget entry")
            if entry["charged_nusd"] != tokens["input_tokens"]*125+tokens["output_tokens"]*500:
                raise ValueError("budget arithmetic differs")
            if entry["charged_nusd"] > entry["reserved_nusd"]:
                raise ValueError("charge exceeded its pre-request reservation")
            docs = [inputs["documents"][key] for key in round_log["nodes_for_match"]]
            docs = [tokenizer.decode(tokenizer.encode(d.replace("Title: Content: ","").strip(),disallowed_special=())[:1024]) for d in docs]
            payload = call["request"]
            if payload["messages"] != messages(query,docs):
                raise ValueError("model did not receive the frozen candidate text/query")
            if payload != dict(model=protocol["model"],messages=messages(query,docs),
                               reasoning_effort=protocol["reasoning_effort"],
                               max_completion_tokens=protocol["max_output_tokens"],store=False,service_tier="default"):
                raise ValueError("model parameters differ from frozen protocol")
            maximum_input = len(json.dumps(payload,ensure_ascii=False).encode())+4096
            if entry["maximum_input_tokens"] != maximum_input or tokens["input_tokens"] > maximum_input:
                raise ValueError("input token bound differs")
            if entry["reserved_nusd"] != maximum_input*125+protocol["max_output_tokens"]*500:
                raise ValueError("pre-request reservation differs")
            if not math.isfinite(call["latency_ms"]) or call["latency_ms"] <= 0:
                raise ValueError("invalid model-call duration")
            permutation(call["response"]["choices"][0]["message"]["content"],len(docs))
    if sorted(indices) != list(range(len(ledger["entries"]))):
        raise ValueError("budget entries are missing, reused or not represented by receipts")
    for method in METHODS:
        values = [relevance(r["ids"],inputs["qrels"][r["query"]],r["candidates"]) for r in records if r["method"] == method]
        for key in values[0]:
            if abs(statistics.mean(r[key] for r in values)-report["methods"][method]["blitzrank"][key]) > 1e-14:
                raise ValueError("reranked summary disagrees with independent relevance oracle")
    destination = recorded/"summary.json"
    text = json.dumps(report,indent=2)+"\n"
    if destination.exists() and destination.read_text() != text:
        raise ValueError("retained summary differs from recomputation")
    if not destination.exists():
        with destination.open("x") as stream:
            stream.write(text)
    print(json.dumps(dict(verified=True,queries=len(protocol["evaluation_ids"]),
                         reranked_requests=len(records),model_calls=len(indices),
                         conservative_usd=report["budget"]["conservative_charge_usd"],
                         summary_sha256=sha(destination))))


if __name__ == "__main__":
    main()
