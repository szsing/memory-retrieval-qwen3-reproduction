"""Primary-seed paired tests; never treat three identical seeds as independent cases."""
from night_data import *
from collections import defaultdict
from scipy.stats import binomtest
P=HERE/'controlled-distractors-v4';rows=json.loads((P/'retrieval-results.json').read_text());primary=[r for r in rows if r['level']==10 and r['seed']==42];G={m:{r['task_id']:r for r in primary if r['method']==m} for m in ['both','write_only','fact_prior','dense']};results={}
for m in ['write_only','fact_prior','dense']:
 a=G['both'];b=G[m];keys=a.keys()&b.keys();win=sum(a[k]['hit6']>b[k]['hit6'] for k in keys);loss=sum(a[k]['hit6']<b[k]['hit6'] for k in keys);blocks=defaultdict(list)
 for k in keys:blocks[a[k]['target_fact']].append(a[k]['hit6']-b[k]['hit6'])
 weights=[sum(v) for v in blocks.values() if sum(v)!=0];observed=abs(sum(weights))
 if len(weights)<=20:
  values=np.array([0.])
  for w in weights:values=np.concatenate([values+w,values-w])
  p=float(np.mean(np.abs(values)>=observed-1e-12))
 else:
  rng=np.random.default_rng(42);null=(rng.choice([-1,1],size=(100000,len(weights)))@np.array(weights));p=(np.count_nonzero(np.abs(null)>=observed)+1)/100001
 results['both vs '+m]={'primary_cases':len(keys),'ours_hit6':sum(a[k]['hit6'] for k in keys),'control_hit6':sum(b[k]['hit6'] for k in keys),'wins':win,'losses':loss,'mcnemar_case_level_exact_unadjusted':binomtest(win,win+loss,.5).pvalue if win+loss else 1.,'target_fact_block_signflip_p_unadjusted':p,'discordant_fact_blocks':len(weights),'warning':'Exploratory unadjusted tests; bootstrap CI with few discordant cases can be optimistic. Seeds are repeated measures, not three independent datasets.'}
dump(P/'retrieval-primary-tests.json',results);print(json.dumps(results))
