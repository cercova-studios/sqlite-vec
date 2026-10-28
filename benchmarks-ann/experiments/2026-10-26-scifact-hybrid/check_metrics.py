"""Cross-check saved ranking metrics against the standard trec_eval implementation."""
from pathlib import Path
import gzip
import importlib.metadata
import json
import math
import pytrec_eval
import evaluate as ev


def main():
    root=Path(__file__).resolve().parent
    receipts=json.loads(gzip.decompress((root/'recorded/receipts.json.gz').read_bytes()))
    checks=0
    for scenario in ('full','scoped'):
        valid={i for i in receipts['document_ids'] if ev.allowed(ev.scope(i,scenario,set()))}
        judgments={q:{i:g for i,g in rels.items() if int(i) in valid and g>0} for q,rels in receipts['judgments'].items()}
        answerable={q:rels for q,rels in judgments.items() if rels}
        oracle=pytrec_eval.RelevanceEvaluator(answerable,{'recall_10','ndcg_cut_10','recip_rank'})
        for method in ('vector','keyword','hybrid'):
            group=[r for r in receipts['results'] if r['scenario']==scenario and r['method']==method]
            ranks={r['query_id']:{str(i):float(10-rank) for rank,i in enumerate(r['ids'])} for r in group}
            expected=oracle.evaluate(ranks)
            for row in group:
                if row['query_id'] not in answerable:
                    assert row['metrics']['recall_at_10'] is None
                    continue
                for metric,key in [('recall_at_10','recall_10'),('ndcg_at_10','ndcg_cut_10'),('reciprocal_rank','recip_rank')]:
                    assert math.isclose(row['metrics'][metric],expected[row['query_id']][key],abs_tol=1e-12)
                    checks+=1
    report=dict(package='pytrec-eval-terrier',version=importlib.metadata.version('pytrec-eval-terrier'),
                metric_comparisons=checks,passed=True,scope='all answerable saved query/method/context points; ranking order preserved with unique rank scores')
    (root/'recorded/metric-oracle.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
