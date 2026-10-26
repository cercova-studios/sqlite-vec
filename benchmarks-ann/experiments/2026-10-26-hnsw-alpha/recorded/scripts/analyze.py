from pathlib import Path
import csv,json,math,statistics
ROOT=Path('/tmp/sqlite-vec-alpha-experiment')
jobs=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.json'))]
assert len(jobs)==24,len(jobs)
lookup={(j['dataset'],j['alpha'],j['seed']):j for j in jobs}
assert len(lookup)==24
builds=json.loads((ROOT/'builds/manifest.json').read_text())
keys=['dataset','alpha','ef_search','recall_at_10','recall_seed_min','recall_seed_max','tie_aware_recall_at_10','qps','qps_seed_min','qps_seed_max','qps_change_percent','qps_change_seed_min_percent','qps_change_seed_max_percent','recall_delta_pp','recall_delta_seed_min_pp','recall_delta_seed_max_pp','p95_us','build_seconds','build_change_percent','peak_rss_mib','neighbor_ids']
summary=[]
for dataset in sorted({j['dataset'] for j in jobs}):
 for alpha in [1.,1.05,1.1,1.2]:
  for ef in [32,64,128,256]:
   selected=[lookup[dataset,alpha,seed] for seed in [0,1,2]]
   baselines=[lookup[dataset,1.,seed] for seed in [0,1,2]]
   configs=[j['configurations'][str(ef)] for j in selected]
   base=[j['configurations'][str(ef)] for j in baselines]
   for j in selected:
    assert j['library_sha256']==builds['variants'][f'alpha-{alpha:g}']['library_sha256']
   for cfg in configs:
    assert len(cfg['rounds'])==5 and cfg['distances_checked']==10000
    assert all(len(r['latencies_us'])==1000 for r in cfg['rounds'])
   recall=[c['recall_at_10'] for c in configs];qps=[c['median_qps'] for c in configs]
   ratios=[c['median_qps']/b['median_qps'] for c,b in zip(configs,base)]
   deltas=[100*(c['recall_at_10']-b['recall_at_10']) for c,b in zip(configs,base)]
   geomean=lambda values:math.exp(statistics.mean(math.log(v) for v in values))
   summary.append(dict(zip(keys,[dataset,alpha,ef,statistics.mean(recall),min(recall),max(recall),statistics.mean(c['tie_aware_recall_at_10'] for c in configs),geomean(qps),min(qps),max(qps),100*(geomean(ratios)-1),100*(min(ratios)-1),100*(max(ratios)-1),statistics.mean(deltas),min(deltas),max(deltas),statistics.median(c['median_p95_us'] for c in configs),statistics.median(j['build_seconds'] for j in selected),100*(geomean(j['build_seconds']/b['build_seconds'] for j,b in zip(selected,baselines))-1),max(j['peak_rss_bytes']/1024**2 for j in selected),statistics.mean(j['stored_graph']['neighbor_ids'] for j in selected)])))
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
with (ROOT/'summary.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(summary)
for row in summary:
 if row['alpha']==1.1:
  print(f"{row['dataset']} ef={row['ef_search']}: recall {row['recall_at_10']:.4f}, delta {row['recall_delta_pp']:+.3f}pp, QPS {row['qps']:.0f}, change {row['qps_change_percent']:+.1f}% (orders {row['qps_change_seed_min_percent']:+.1f}% to {row['qps_change_seed_max_percent']:+.1f}%)",flush=True)
print('Verified 24 builds, 96 quality points, 480 timing rounds, 480000 timed queries and 960000 returned-distance checks.')
