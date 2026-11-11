"""Binary SciFact relevance metrics and paired query bootstrap, independent of scoring."""
import math
import random
import statistics


def metrics(ids, judgments, candidates=None):
    if len(ids) > 10 or len(ids) != len(set(ids)):
        raise ValueError("unique top-ten IDs required")
    positives = {int(key) for key, value in judgments.items() if value > 0}
    if not positives:
        raise ValueError("query needs at least one labelled relevant document")
    gains = [1 if key in positives else 0 for key in ids]
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10,len(positives))))
    return dict(ndcg10=sum(g / math.log2(i+2) for i,g in enumerate(gains)) / ideal,
                mrr10=next((1/(i+1) for i,g in enumerate(gains) if g),0.0),
                recall10=sum(gains)/len(positives),
                candidate_recall=len(set(candidates or ids)&positives)/len(positives))


def bootstrap(differences, samples=10000, seed=20261110):
    if not differences or samples < 1:
        raise ValueError("paired observations and positive sample count required")
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n for _ in range(samples))
    return dict(mean=statistics.mean(differences), low=means[int(.025*(samples-1))],
                high=means[int(.975*(samples-1))], samples=samples, seed=seed)


def percentile(values, p):
    if not values:
        raise ValueError("nonempty measurements required")
    return sorted(values)[max(0,math.ceil(p*len(values))-1)]
