"""Frozen multi-fact synthetic-profile stress test; development-only donor facts."""
from night_data import *
from night_worker import load_case
from night_methods import props,vec,match,rrf,ranked
from night_budget import NightTransport
from concurrent.futures import ThreadPoolExecutor
import argparse,collections
OUT=HERE/'controlled-distractors-v1';OUT.mkdir(exist_ok=True)
T=NightTransport('controlled-negative-construction')
METHODS=['both','write_only','fact_prior','dense'];SEEDS=[42,43,44];LEVELS=[5,10,20]
def atomic(name,v):dump(OUT/name,v)
def pool():
 tasks,_,_=load_data();dev=set(dev_ids(tasks));donors=[];seen=set()
 for t in sorted(tasks,key=lambda t:str(t['task_id'])):
  if t['task_id'] not in dev or t['user_message'] in seen:continue
  seen.add(t['user_message']);p=props(t['user_message'],True)
  if not p:continue
  donors.append({'id':len(donors),'source_task_id':t['task_id'],'fact':t['user_message'],'properties':p,'messages':json.dumps([{'role':'user','content':t['user_message']},{'role':'assistant','content':t['assistant_message']}]),'text':text([{'role':'user','content':t['user_message']},{'role':'assistant','content':t['assistant_message']}])})
 atomic('development-donor-pool.json',donors);return donors
PAIR_PROMPT='''Audit synthetic profile consistency. The facts below would be used as statements by ONE fictional user at the same time. Identify only pairs that directly contradict each other or cannot plausibly be simultaneously true. Coexisting medical conditions, restrictions, and preferences are not contradictions. Equivalent duplicate facts are also incompatible for this test. Do not select on retrieval performance. Return JSON {"conflict_pairs": [[id,id],...], "note": "brief caveat"}.
FACTS:\n'''
AUDIT_PROMPT='''Construct negative memories for a controlled retrieval test, not an answer to the user.
A synthetic user has the TARGET FACT and asks the QUERY. For each candidate fact, decide whether it can be used as an IRRELEVANT, COMPATIBLE distractor for this query. Accept only if (1) it can coexist with the target fact, (2) it neither repeats/paraphrases the target nor supplies the target's decisive information, and (3) it does not independently impose a material constraint on an appropriate answer to the query. Reject potentially relevant facts, including relevant contraindications or eligibility constraints, even if they are from a different topic. When uncertain reject. Do not use embedding scores, properties, or any method output.
Return JSON with eligible_ids and rejections [{"id":...,"reason":"relevant"|"conflict"|"duplicate"|"uncertain"}]. Every candidate ID must appear exactly once across these lists. No explanation or answer generation.
'''
def construct():
 tasks,_,_=load_data();dev=set(dev_ids(tasks));test=[t for t in tasks if t['task_id'] not in dev];donors=pool();assert len(test)==85
 spec={'experiment':'synthetic-profile controlled multi-fact distractors, not original InMind scores','test_task_ids':[t['task_id'] for t in test],'donor_source':'development 40 only; exact duplicate facts deduplicated; nonempty original write-side properties required','levels':LEVELS,'seeds':SEEDS,'primary_level':10,'methods':METHODS,'rrf':60,'property_weight':2,'retrieval_k':6,'context_token_budget':4096,'selection':'query/target compatibility annotation then uniform seeded greedy sampling; never retrieval scores','negative_annotation':'local Qwen; not human ground truth; rejection on uncertain labels','missing_eligible_policy':'report coverage; never silently drop cases or resample based on retrieval performance','primary_population':'all85; prespecified mechanistic stratum target_has_write_properties','paired_clusters':'exact target-fact string; seeds are repeated measures, not independent samples','source_sha256':sha(__file__),'night_methods_sha256':sha(HERE/'night_methods.py'),'model':MODEL}
 lock=OUT/'protocol.lock.json'
 if lock.exists():assert json.loads(lock.read_text())==spec,'Protocol changed: use a new experiment version'
 else:atomic('protocol.lock.json',spec)
 pairfile=OUT/'global-conflicts.json'
 if pairfile.exists():pairdata=json.loads(pairfile.read_text())
 else:
  schema={'type':'object','properties':{'conflict_pairs':{'type':'array','items':{'type':'array','minItems':2,'maxItems':2,'items':{'type':'integer','minimum':0,'maximum':len(donors)-1}}},'note':{'type':'string'}},'required':['conflict_pairs','note']}
  pairdata=T.call([{'role':'user','content':PAIR_PROMPT+json.dumps([{'id':d['id'],'fact':d['fact']} for d in donors])}],schema=schema,json_output=True,max_tokens=2048);atomic('global-conflicts.json',pairdata)
 forbidden={frozenset(p) for p in pairdata['conflict_pairs']}
 def one(t):
  path=OUT/'annotations'/(str(t['task_id'])+'.json');path.parent.mkdir(exist_ok=True)
  if path.exists():return json.loads(path.read_text())
  candidates=[d for d in donors if d['fact']!=t['user_message']];ids=[d['id'] for d in candidates]
  schema={'type':'object','properties':{'eligible_ids':{'type':'array','items':{'enum':ids}},'rejections':{'type':'array','items':{'type':'object','properties':{'id':{'enum':ids},'reason':{'enum':['relevant','conflict','duplicate','uncertain']}},'required':['id','reason']}}},'required':['eligible_ids','rejections']}
  payload={'TARGET_FACT':t['user_message'],'QUERY':t['query'],'CANDIDATES':[{'id':d['id'],'fact':d['fact']} for d in candidates]}
  audit=T.call([{'role':'user','content':AUDIT_PROMPT+json.dumps(payload)}],schema=schema,json_output=True,max_tokens=2048)
  observed=audit['eligible_ids']+[r['id'] for r in audit['rejections']]
  assert len(observed)==len(ids) and set(observed)==set(ids),'Incomplete/duplicated negative annotations'
  sets={}
  for seed in SEEDS:
   rng=np.random.default_rng(int(hashlib.sha256((str(t['task_id'])+'|'+str(seed)).encode()).hexdigest()[:16],16));order=list(sorted(audit['eligible_ids']));rng.shuffle(order);chosen=[]
   for i in order:
    if not any(frozenset([i,j]) in forbidden for j in chosen):chosen.append(i)
   sets[str(seed)]=chosen
  result={'task_id':t['task_id'],'target_fact':t['user_message'],'query':t['query'],'audit':audit,'seeded_compatible_order':sets}
  dump(path,result);event('negative_annotation_done',task=t['task_id'],eligible=len(audit['eligible_ids']),compatible_min=min(map(len,sets.values())))
  return result
 with ThreadPoolExecutor(max_workers=4) as executor:annotations=list(executor.map(one,test))
 coverage={str(k):{str(s):sum(len(a['seeded_compatible_order'][str(s)])>=k for a in annotations) for s in SEEDS} for k in LEVELS}
 atomic('construction-audit.json',{'donor_facts':len(donors),'test_cases':85,'coverage':coverage,'negative_annotation_calls':T.calls,'negative_annotation_cache_hits':T.cached,'conflict_pairs':pairdata['conflict_pairs'],'all_positive_properties_checked':True,'all_donor_task_ids_are_development':all(d['source_task_id'] in dev for d in donors),'retrieval_has_not_run_at_construction_end':not (OUT/'retrieval-results.json').exists()})
 return test,donors,{str(a['task_id']):a for a in annotations}
def retrieve():
 test,donors,annotations=construct();results=[];old={}
 out=OUT/'retrieval-results.json'
 if out.exists():old={(str(r['task_id']),r['level'],r['seed'],r['method']):r for r in json.loads(out.read_text())}
 donorby={d['id']:d for d in donors}
 with connect() as c:original={(str(cid),m):json.loads(v) for cid,m,qid,v in c.execute("select id,method,qid,value from results where dataset='inmind' and qid='query'")}
 for t in test:
  cid=str(t['task_id']);base,_=load_case('inmind',cid);bp=[props(n['user_text'],True) for n in base];qv=vec(props(t['query']));qe=vec([t['query']])[0]
  target=next(i for i,n in enumerate(base) if t['user_message'] in n['text'] and t['assistant_message'] in n['text']);target_id=base[target]['id'];ann=annotations[cid]
  configs=[(0,42)]+[(k,s) for k in LEVELS for s in SEEDS if len(ann['seeded_compatible_order'][str(s)])>=k]
  for k,seed in configs:
   if all((cid,k,seed,m) in old for m in METHODS):results.extend(old[cid,k,seed,m] for m in METHODS);continue
   ds=[donorby[j] for j in ann['seeded_compatible_order'][str(seed)][:k]];assert all(d['properties'] for d in ds)
   nd=[{'id':'donor-'+str(d['id']),'text':d['text'],'properties':d['properties'],'is_donor':True} for d in ds]
   corpus=[{'id':n['id'],'text':n['text'],'properties':p,'is_donor':False} for n,p in zip(base,bp)]
   if k:
    rng=np.random.default_rng(int(hashlib.sha256(('position|'+cid+'|'+str(seed)).encode()).hexdigest()[:16],16))
    # Insert distractors randomly, preserving the original background/target relative order.
    slots=rng.integers(0,len(base)+1,len(nd));combined=[]
    for pos in range(len(base)+1):
     combined.extend(n for j,n in enumerate(nd) if slots[j]==pos)
     if pos<len(base):combined.append(corpus[pos])
    corpus=combined
   assert len(corpus)==len(base)+k and sum(n['is_donor'] for n in corpus)==k
   E=vec([n['text'] for n in corpus]);lv=[vec(n['properties']) for n in corpus];dense=E@qe
   semantic=match(lv,qv);write=match(lv,qe[None,:]);prior=np.array([int(v is not None) for v in lv]);scores={'both':rrf(dense)+2*rrf(semantic),'write_only':rrf(dense)+2*rrf(write),'fact_prior':rrf(dense)+2*rrf(prior),'dense':dense}
   targetpos=next(i for i,n in enumerate(corpus) if n['id']==target_id);assert not corpus[targetpos]['is_donor']
   for method,score in scores.items():
    order=ranked(score);ctx=budget_context([corpus[i]['text'] for i in order[:6]]);rank=order.index(targetpos)+1
    if k==0:assert ctx==original[cid,method]['context'],('zero-distractor baseline changed',cid,method)
    r={'task_id':t['task_id'],'level':k,'seed':seed,'method':method,'target_fact':t['user_message'],'query':t['query'],'gold':t,'target_has_properties':bool(bp[target]),'query_has_properties':qv is not None,'candidate_count':len(corpus),'nonempty_property_count':sum(v is not None for v in lv),'donor_ids':[d['id'] for d in ds],'target_rank':rank,'hit1':int(rank<=1),'hit3':int(rank<=3),'hit6':int(rank<=6),'mrr':1/rank,'top6_note_ids':[corpus[i]['id'] for i in order[:6]],'context':ctx,'context_tokens':token_count(ctx),'context_has_literal_target':t['user_message'] in ctx}
    results.append(r)
  atomic('retrieval-results.json',results);event('controlled_retrieval_case',task=t['task_id'],rows=len(results))
 atomic('retrieval-complete.json',{'test_cases':85,'rows':len(results),'source_sha256':sha(__file__)});summarize(results)
def summarize(rows=None):
 rows=rows or json.loads((OUT/'retrieval-results.json').read_text());summary={};paired={}
 for k in [0]+LEVELS:
  for stratum in ['all85','target_has_properties']:
   selected=[r for r in rows if r['level']==k and (stratum=='all85' or r['target_has_properties'])]
   key=str(k)+'/'+stratum;summary[key]={}
   for m in METHODS:
    rr=[r for r in selected if r['method']==m];n=len(rr)
    if n:summary[key][m]={'case_seed_pairs':n,'unique_tasks':len({r['task_id'] for r in rr}),'hit6_count':sum(r['hit6'] for r in rr),'hit6':sum(r['hit6'] for r in rr)/n,'hit1':sum(r['hit1'] for r in rr)/n,'mrr':sum(r['mrr'] for r in rr)/n,'context_target_literal':sum(r['context_has_literal_target'] for r in rr)/n,'population_denominator':85 if stratum=='all85' else 41}
   ours={(r['task_id'],r['seed']):r for r in selected if r['method']=='both'}
   for m in ['write_only','fact_prior','dense']:
    other={(r['task_id'],r['seed']):r for r in selected if r['method']==m};keys=ours.keys()&other.keys()
    if not keys:continue
    groups=collections.defaultdict(list)
    for i in keys:groups[ours[i]['target_fact']].append(ours[i]['hit6']-other[i]['hit6'])
    cs=sorted(groups);sums=np.array([sum(groups[c]) for c in cs]);sizes=np.array([len(groups[c]) for c in cs]);rng=np.random.default_rng(42);boot=[]
    for _ in range(2000):
     ii=rng.integers(0,len(cs),len(cs));boot.append(sums[ii].sum()/sizes[ii].sum())
    paired[key+' both vs '+m]={'paired_case_seeds':len(keys),'unique_tasks':len({i[0] for i in keys}),'fact_clusters':len(cs),'delta_hit6':float(sums.sum()/sizes.sum()),'cluster_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'same_context':sum(ours[i]['context']==other[i]['context'] for i in keys),'wins':sum(ours[i]['hit6']>other[i]['hit6'] for i in keys),'losses':sum(ours[i]['hit6']<other[i]['hit6'] for i in keys),'limitation':'Synthetic dev-donor stress test, local-model negative annotations; repeated seeds clustered by target fact. Coverage can be below85 if compatible donors insufficient.'}
 atomic('retrieval-summary.json',summary);atomic('retrieval-paired.json',paired)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['construct','retrieve','summary'],default='retrieve');a=p.parse_args()
 if a.phase=='construct':construct()
 elif a.phase=='retrieve':retrieve()
 else:summarize()
