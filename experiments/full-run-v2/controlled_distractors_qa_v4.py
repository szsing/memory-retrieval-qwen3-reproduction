"""End-to-end application test, all eligible primary 10-donor cases, three methods."""
from night_data import *
from night_budget import NightTransport
from concurrent.futures import ThreadPoolExecutor,as_completed
from scipy.stats import binomtest
import collections,traceback
OUT=HERE/'controlled-distractors-v4';T=NightTransport('controlled-distractor-reader-judge')
METHODS=['both','write_only','fact_prior'];SCORE={'type':'object','properties':{'score':{'type':'integer','enum':[0,1]},'reason':{'type':'string'}},'required':['score','reason']}
def run():
 assert (OUT/'retrieval-complete.json').exists()
 rows=[r for r in json.loads((OUT/'retrieval-results.json').read_text()) if r['level']==10 and r['seed']==42 and r['method'] in METHODS]
 assert len({(r['task_id'],r['method']) for r in rows})==len(rows)
 protocol={'primary_distractors':10,'primary_seed':42,'methods':METHODS,'population_denominator':85,'eligible_tasks':len({r['task_id'] for r in rows}),'reader':MODEL,'reader_max_tokens':4096,'judge':MODEL,'judge_max_tokens':2048,'prompts':'unchanged official InMind answer_system, judge_application, judge_target_recall','purpose':'same-backbone controlled mechanism diagnostic, not official full benchmark score or independent human validation','context_token_budget':4096,'source_sha256':sha(__file__),'retrieval_protocol_sha256':sha(OUT/'protocol.lock.json')}
 lock=OUT/'qa-protocol.lock.json'
 if lock.exists():assert json.loads(lock.read_text())==protocol
 else:dump(lock,protocol)
 (OUT/'qa').mkdir(exist_ok=True);prompts=VENDOR/'InMind-main/evaluation/prompts'
 def one(r):
  path=OUT/'qa'/(str(r['task_id'])+'__'+r['method']+'.json')
  if path.exists():return json.loads(path.read_text())
  try:
   t=r['gold'];ctx=r['context'];answer=T.call([{'role':'system','content':(prompts/'answer_system.txt').read_text().replace('{context}',ctx)},{'role':'user','content':r['query']}],max_tokens=4096)
   payload={k:t[k] for k in ['user_message','explanation']};payload.update(context=ctx,query=r['query'],answer=answer)
   app=T.call([{'role':'system','content':(prompts/'judge_application.txt').read_text()},{'role':'user','content':json.dumps(payload)}],schema=SCORE,json_output=True,max_tokens=2048)
   payload.pop('answer');recall=T.call([{'role':'system','content':(prompts/'judge_target_recall.txt').read_text()},{'role':'user','content':json.dumps(payload)}],schema=SCORE,json_output=True,max_tokens=2048)
   result={'task_id':r['task_id'],'method':r['method'],'target_fact':r['target_fact'],'target_has_properties':r['target_has_properties'],'level':10,'seed':42,'context_sha256':hashlib.sha256(ctx.encode()).hexdigest(),'answer':answer,'application':app,'target_recall':recall}
   dump(path,result);error=path.with_suffix('.error.json')
   if error.exists():error.unlink()
   return result
  except Exception:
   dump(path.with_suffix('.error.json'),{'task_id':r['task_id'],'method':r['method'],'error':traceback.format_exc()});raise
 # Deterministic interleaving avoids completing all of one method first.
 rows.sort(key=lambda r:(int(hashlib.sha256(str(r['task_id']).encode()).hexdigest()[:8],16),METHODS.index(r['method'])))
 failures=[]
 with ThreadPoolExecutor(max_workers=4) as executor:
  fs={executor.submit(one,r):r for r in rows}
  for i,f in enumerate(as_completed(fs),1):
   r=fs[f]
   try:f.result()
   except Exception:failures.append([r['task_id'],r['method']])
   event('controlled_qa_progress',done=i,total=len(rows),task=r['task_id'],method=r['method'],failures=len(failures),uncached_calls=T.calls,cache_hits=T.cached)
 summarize()
 dump(OUT/'qa-run-finished.json',{'expected':len(rows),'population_denominator':85,'failures':failures,'uncached_calls':T.calls,'cache_hits':T.cached,'all_eligible_finished':not failures})
def summarize():
 files=[f for f in (OUT/'qa').glob('*.json') if not f.name.endswith('.error.json')];rows=[json.loads(f.read_text()) for f in files];summary={};paired={}
 for stratum in ['all85','target_has_properties']:
  rr=[r for r in rows if stratum=='all85' or r['target_has_properties']];groups={m:{r['task_id']:r for r in rr if r['method']==m} for m in METHODS}
  summary[stratum]={m:{'n':len(v),'population_denominator':85 if stratum=='all85' else 41,'application_correct':sum(r['application']['score'] for r in v.values()),'target_recall_correct':sum(r['target_recall']['score'] for r in v.values())} for m,v in groups.items()}
  for m in ['write_only','fact_prior']:
   a=groups['both'];b=groups[m];ks=a.keys()&b.keys()
   if not ks:continue
   cl=collections.defaultdict(list)
   for k in ks:cl[a[k]['target_fact']].append(a[k]['application']['score']-b[k]['application']['score'])
   cs=sorted(cl);sums=np.array([sum(cl[c]) for c in cs]);sizes=np.array([len(cl[c]) for c in cs]);rng=np.random.default_rng(42);boot=[]
   for _ in range(2000):
    ii=rng.integers(0,len(cs),len(cs));boot.append(sums[ii].sum()/sizes[ii].sum())
   wins=sum(a[k]['application']['score']>b[k]['application']['score'] for k in ks);losses=sum(a[k]['application']['score']<b[k]['application']['score'] for k in ks)
   paired[stratum+' both vs '+m]={'n':len(ks),'complete_all85':stratum=='all85' and len(ks)==85,'ours_correct':sum(a[k]['application']['score'] for k in ks),'control_correct':sum(b[k]['application']['score'] for k in ks),'delta_application':float(sums.sum()/sizes.sum()),'cluster_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'wins':wins,'losses':losses,'mcnemar_exact_unadjusted':binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.,'limitation':'Synthetic standing-rule profiles and same-model evaluator; no generalization beyond this controlled test.'}
 dump(OUT/'qa-summary.json',summary);dump(OUT/'qa-paired.json',paired)
if __name__=='__main__':run()
