from pathlib import Path
import json,os,html
ROOT=Path('/tmp/sqlite-vec-alpha-experiment')
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl-cache')
import matplotlib
config=ROOT/'fontconfig.xml'
config.write_text('<fontconfig><dir>'+html.escape(str(Path(matplotlib.get_data_path())/'fonts/ttf'))+'</dir><cachedir>'+str(ROOT/'font-cache')+'</cachedir></fontconfig>')
os.environ['FONTCONFIG_FILE']=str(config)

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path('/tmp/sqlite-vec-alpha-experiment');rows=json.loads((ROOT/'summary.json').read_text());paired=json.loads((ROOT/'paired_summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
styles={1.0:('black','o'),1.05:('#0072B2','s'),1.1:('#D55E00','^'),1.2:('#009E73','D')}
for ax,dataset,title in zip(axes,['glove-100-angular','sift-128-euclidean'],['GloVe-100 / cosine','SIFT-128 / L2']):
 for alpha,(color,marker) in styles.items():
  selected=sorted([r for r in rows if r['dataset']==dataset and r['alpha']==alpha],key=lambda r:r['ef_search'])
  x=np.array([r['recall_at_10']*100 for r in selected]);y=np.array([r['qps'] for r in selected])
  xe=np.maximum(np.array([[r['recall_at_10']-r['recall_seed_min'] for r in selected],[r['recall_seed_max']-r['recall_at_10'] for r in selected]])*100,0)
  ye=np.maximum(np.array([[r['qps']-r['qps_seed_min'] for r in selected],[r['qps_seed_max']-r['qps'] for r in selected]]),0)
  ax.errorbar(x,y,xerr=xe,yerr=ye,color=color,marker=marker,label=f'alpha={alpha:g}',capsize=2,linewidth=1.3,markersize=5)
  if alpha==1:
   for r,xi,yi in zip(selected,x,y):ax.annotate(f"ef={r['ef_search']}",(xi,yi),xytext=((-48,-12) if r['ef_search']>=128 else (4,5)),textcoords='offset points',fontsize=8)
 ax.set(title=title,xlabel='Strict recall@10 (%)',ylabel='Queries / second (warm SQL)',yscale='log');ax.grid(True,alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Experimental HNSW alpha sweep: 50,000 real vectors per dataset',fontsize=13)
fig.supxlabel('M=16; ef_construction=200; ef_search=32/64/128/256; 1,000 queries; three insertion orders\nQPS: geometric mean of per-order medians over five rounds. Error bars: observed order range, not confidence intervals.',fontsize=9)
for suffix in ['png','svg','pdf']:fig.savefig(ROOT/f'recall-qps.{suffix}',dpi=160)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True,layout='constrained')
for ax,dataset,title in zip(axes,['glove-100-angular','sift-128-euclidean'],['GloVe-100 / cosine','SIFT-128 / L2']):
 selected=sorted([r for r in paired if r['dataset']==dataset],key=lambda r:r['ef_search']);y=np.array([r['qps_change_percent'] for r in selected]);lo=np.array([r['timing_only_bootstrap_ci_percent'][0] for r in selected]);hi=np.array([r['timing_only_bootstrap_ci_percent'][1] for r in selected])
 ax.errorbar(range(4),y,yerr=[y-lo,hi-y],fmt='o',capsize=5,color='#0072B2');ax.axhline(0,color='black',linewidth=1);ax.set_xticks(range(4),[r['ef_search'] for r in selected]);ax.set(title=title,xlabel='ef_search');ax.grid(True,alpha=.2)
axes[0].set_ylabel('QPS change: alpha=1.1 versus 1.0 (%)')
fig.suptitle('Alternating timing confirmation: alpha=1.1',fontsize=13)
fig.supxlabel('Ten paired rounds, seed-0 graph, identical query orders. 95% bootstrap intervals cover timing rounds only;\nthey do not cover other graphs, datasets, hardware or production workloads.',fontsize=9)
for suffix in ['png','svg','pdf']:fig.savefig(ROOT/f'paired-confirmation.{suffix}',dpi=160)
