"""Paired cluster bootstrap; official denominator retained even when only partial runs exist."""
from night_data import *
from collections import defaultdict
from scipy.stats import binomtest

def main():
 with connect() as c:
  rows=c.execute('select dataset,id,method,qid,value from results').fetchall()
  expected=defaultdict(int)
  for ds,cid,qs in c.execute('select dataset,id,queries from cases'):
   for q in json.loads(qs):
    if ds=='inmind' and (q['id']!='query' or not q['heldout']):continue
    expected[ds]+=1
 groups=defaultdict(dict)
 for ds,cid,m,qid,raw in rows:
  v=json.loads(raw)
  if 'evaluation' not in v:continue
  if ds=='inmind' and (qid!='query' or not v['heldout']):continue
  if ds=='inmind':cluster=hashlib.sha256(v['gold']['user_message'].encode()).hexdigest()
  elif ds.startswith('lucid_'):cluster=v['gold']['query_id']
  else:cluster=cid
  groups[ds,m][cid,qid]=(v['evaluation']['score'],cluster)
 result={}
 for ds in expected:
  ours=groups[ds,'both']
  for m in ['dense','hyde','dense_pick','mem0','amem','rootmem','rootmem_both','write_only','query_only','shuffled','fact_prior','gated_fact_prior']:
   other=groups[ds,m];keys=ours.keys()&other.keys()
   if not keys:continue
   bycluster=defaultdict(list)
   win=loss=0
   for k in keys:
    a,cl=ours[k];b,_=other[k];bycluster[cl].append(a-b);win+=int(a>b);loss+=int(a<b)
   clusters=sorted(bycluster);sums=np.array([sum(bycluster[x]) for x in clusters]);sizes=np.array([len(bycluster[x]) for x in clusters]);rng=np.random.default_rng(42);samples=[]
   for _ in range(2000):
    ii=rng.integers(0,len(clusters),len(clusters));samples.append(sums[ii].sum()/sizes[ii].sum())
   result[ds+' both vs '+m]={'paired_n':len(keys),'full_denominator':expected[ds],'complete_pair':len(keys)==expected[ds],'clusters':len(clusters),'ours_correct':sum(ours[k][0] for k in keys),'baseline_correct':sum(other[k][0] for k in keys),'paired_accuracy_difference':float(sums.sum()/sizes.sum()),'cluster_bootstrap_ci95':np.quantile(samples,[.025,.975]).tolist(),'wins':win,'losses':loss,'mcnemar_exact_unadjusted':binomtest(win,win+loss,.5).pvalue if win+loss else 1.,'caveat':'Partial intersections may be biased by completion time; no full-benchmark inference until complete_pair=true. p-values are exploratory, unadjusted across methods; same-model judge requires independent validation.'}
 dump(HERE/'night-paired-analysis.json',result)
if __name__=='__main__':main()
