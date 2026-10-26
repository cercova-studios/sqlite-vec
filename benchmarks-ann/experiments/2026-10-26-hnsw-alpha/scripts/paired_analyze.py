from pathlib import Path
import json,math,numpy as np
from paths import ROOT;records=json.loads((ROOT/'paired_timing.json').read_text());assert len(records)==80
summary=[]
for dataset in ['glove-100-angular','sift-128-euclidean']:
 for ef in [32,64,128,256]:
  selected=[p for p in records if p['dataset']==dataset and p['ef']==ef];assert len(selected)==10
  ratios=[];qps={1.0:[],1.1:[]}
  for pair in selected:
   measured={m['alpha']:m for m in pair['measurements']};assert set(measured)=={1.,1.1}
   for alpha,m in measured.items():
    assert len(m['latencies_us'])==1000;qps[alpha].append(m['qps'])
   ratios.append(measured[1.1]['qps']/measured[1.0]['qps'])
  logs=np.log(ratios);rng=np.random.default_rng(42);bootstrap=np.exp(logs[rng.integers(0,10,size=(10000,10))].mean(1));ci=np.percentile(bootstrap,[2.5,97.5])
  summary.append({'dataset':dataset,'ef_search':ef,'alpha':1.1,'insertion_seed':0,'paired_rounds':10,'qps_change_percent':100*(math.exp(float(logs.mean()))-1),'timing_only_bootstrap_ci_percent':(100*(ci-1)).tolist(),'ratio_minmax_percent':[100*(min(ratios)-1),100*(max(ratios)-1)],'baseline_median_qps':float(np.median(qps[1.0])),'candidate_median_qps':float(np.median(qps[1.1]))})
(ROOT/'paired_summary.json').write_text(json.dumps(summary,indent=2))
print('80 paired rounds validated; timing summary saved.')
