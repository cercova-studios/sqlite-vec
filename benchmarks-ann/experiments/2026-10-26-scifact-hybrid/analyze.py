"""Compute paired relevance differences and candidate coverage from saved receipts."""
from pathlib import Path
import gzip
import json
import numpy as np


def main():
    root=Path(__file__).resolve().parent
    raw=json.loads(gzip.decompress((root/'recorded/receipts.json.gz').read_bytes()))
    output=[]
    for scenario in ('full','scoped'):
        groups={m:{r['query_id']:r for r in raw['results'] if r['scenario']==scenario and r['method']==m}
                for m in ('vector','keyword','hybrid')}
        query_ids=sorted((q for q in groups['hybrid'] if groups['hybrid'][q]['metrics']['relevant']),key=int)
        for baseline in ('vector','keyword'):
            differences=np.array([groups['hybrid'][q]['metrics']['ndcg_at_10']-groups[baseline][q]['metrics']['ndcg_at_10'] for q in query_ids])
            rng=np.random.default_rng(42)
            resamples=differences[rng.integers(0,len(differences),size=(10000,len(differences)))].mean(1)
            output.append(dict(scenario=scenario,baseline=baseline,answerable_queries=len(query_ids),
                mean_ndcg_delta=float(differences.mean()),paired_query_bootstrap_95_ci=np.percentile(resamples,[2.5,97.5]).tolist(),
                wins=int((differences>1e-12).sum()),ties=int((np.abs(differences)<=1e-12).sum()),losses=int((differences< -1e-12).sum())))
    (root/'recorded/paired-quality.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
