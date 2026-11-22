"""Frozen paired comparisons, clustered by identical personal-fact text."""
from common import *
tasks,_,_=load_data();byid={t['task_id']:t for t in tasks};dev=set(dev_ids(tasks));recipe=json.loads((HERE/'retrieval/recipe.lock.json').read_text());selected=recipe['selected'];rows=json.loads((HERE/'retrieval/all125-results.json').read_text())
values={(r['method'],r['kind'],r['task_id']):r['hit_at_6'] for r in rows}
groups={'all125':[t['task_id'] for t in tasks],'disjoint85':[t['task_id'] for t in tasks if t['task_id'] not in dev],'cross_domain61':[t['task_id'] for t in tasks if t['domain'] not in {'health_and_wellness','legal','consumer','personal_development','other'}]}
out={}
for group,ids in groups.items():
 table={m:{k:sum(values[(m,k,i)] for i in ids) for k in ['query','naive_query']} for m in ['dense','hyde',selected]}
 clusters={}
 for i in ids:clusters.setdefault(byid[i]['user_message'],[]).append(i)
 sizes=np.array([len(ii) for ii in clusters.values()]);rng=np.random.default_rng(42);draw=rng.integers(0,len(sizes),size=(10000,len(sizes)))
 comparisons={}
 for base in ['dense','hyde']:
  differences=np.array([sum(values[(selected,'query',i)]-values[(base,'query',i)] for i in ii) for ii in clusters.values()])
  samples=differences[draw].sum(1)/sizes[draw].sum(1)
  comparisons[base]={'observed_delta':float(differences.sum()/len(ids)),'cluster_bootstrap_95':np.quantile(samples,[.025,.975]).tolist(),'cluster_bootstrap_97_5_for_two_planned_comparisons':np.quantile(samples,[.0125,.9875]).tolist(),'rescued':sum(values[(selected,'query',i)]==1 and values[(base,'query',i)]==0 for i in ids),'lost':sum(values[(selected,'query',i)]==0 and values[(base,'query',i)]==1 for i in ids)}
 out[group]={'n':len(ids),'unique_fact_clusters':len(clusters),'hits':table,'paired_indirect_comparisons':comparisons}
dump(HERE/'retrieval/paired-analysis.json',out)
event('paired_analysis_complete',selected=selected)
