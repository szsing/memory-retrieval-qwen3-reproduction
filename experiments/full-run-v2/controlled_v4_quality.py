from night_data import *
from collections import defaultdict
P=HERE/'controlled-distractors-v4'
rows=json.loads((P/'retrieval-results.json').read_text());groups=defaultdict(list)
for r in rows:groups[r['task_id'],r['level'],r['seed']].append(r)
assert len(rows)==2284,len(rows)
for key,rr in groups.items():
 assert len(rr)==4 and set(r['method'] for r in rr)=={'both','write_only','fact_prior','dense'}
 assert len({tuple(r['donor_ids']) for r in rr})==1
 for r in rr:
  assert r['candidate_count']==247+r['level']
  assert r['nonempty_property_count']==r['level']+int(r['target_has_properties'])
  assert r['context_tokens']<=4096
  assert len(r['donor_ids'])==r['level']
base=[r for r in rows if r['level']==0]
assert {m:sum(r['hit6'] for r in base if r['method']==m) for m in ['both','write_only','fact_prior','dense']}=={'both':47,'write_only':47,'fact_prior':47,'dense':7}
summary={}
for k in [0,5,10]:
 for seed in ([42] if k==0 else [42,43,44]):
  selected=[r for r in rows if r['level']==k and r['seed']==seed]
  summary[str(k)+'/'+str(seed)]={m:{'n':len(rr:=[r for r in selected if r['method']==m]),'hit6':sum(r['hit6'] for r in rr),'hit1':sum(r['hit1'] for r in rr),'mean_target_rank':sum(r['target_rank'] for r in rr)/len(rr)} for m in ['both','write_only','fact_prior','dense']}
result={'passed':True,'rows':len(rows),'zero_distractor_all85_exact_baseline_reproduced':True,'same_candidate_pools':True,'all_distractors_have_properties':True,'k_plus_target_nonempty_count_asserted':True,'context_budget_asserted':True,'per_seed_results':summary}
dump(P/'quality-check.json',result);print(json.dumps(result))
