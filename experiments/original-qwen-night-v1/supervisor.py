"""Eight-hour durable source-matched queue, with a preregistered full-expansion gate."""
import os,sys,json,time,subprocess,hashlib,collections,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
from instrument import budget,BudgetExpired

def dump(n,x):
 p=HERE/n;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
def diagnose(prefix='subset-',exclude_legacy=False):
 c=collections.Counter()
 if (HERE/'events.jsonl').exists():
  for line in (HERE/'events.jsonl').open():
   r=json.loads(line)
   if not r.get('job','').startswith(prefix) or (exclude_legacy and 'legacy_' in r.get('job','')):continue
   k=r['kind']
   if k=='api_response':c['api_calls']+=1;c['truncated']+=r.get('finish_reason')=='length';c['thinking_content']+=bool(r.get('thinking_text') or r.get('reasoning_content'))
   if k=='jcall_return':c['jcall_calls']+=1;c['jcall_none']+=r['returned_none']
   if k=='json_text_check':c['json_text_attempts']+=1;c['invalid_json_attempts']+=not r['valid_json']
   if k=='by_id':c['expected_items']+=r['expected'];c['missing_items']+=r['missing'];c['empty_lists']+=r['empty']
   if k=='methods_batched_return':c['batch_return_items']+=r['n'];c['batch_none_items']+=r['returned_none']
 return dict(c)
def run(spec):
 name=spec['name'];p=HERE/'results'/(name+'.json')
 if p.exists():return True
 budget();state['current_job']=name;dump('status.json',state);log=HERE/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
 env=dict(os.environ,RUN_JOB=name)
 with log.open('a') as f:code=subprocess.run([sys.executable,str(HERE/'job.py'),json.dumps(spec)],stdout=f,stderr=subprocess.STDOUT,env=env).returncode
 if code==75:raise BudgetExpired()
 state['completed' if code==0 else 'failed'].append(name);dump('status.json',state)
 return code==0

def gate():
 comparisons={};passed=True
 for bed,key in [('inmind','inmind_off'),('lucid','lucid_test'),('implex','implex_test')]:
  rows={}
  for m in ['plain','hyde','full','rerank']:
   p=HERE/'results'/('subset-'+bed+'-'+m+'.json')
   if not p.exists():passed=False;continue
   rows[m]=json.loads(p.read_text())['result']
  if len(rows)!=4:continue
  v={m:rows[m][key] for m in rows};n=len(v['full']);assert n=={'inmind':61,'lucid':240,'implex':81}[bed]
  wins=sum(a>b for a,b in zip(v['full'],v['hyde']));losses=sum(a<b for a,b in zip(v['full'],v['hyde']));delta=(sum(v['full'])-sum(v['hyde']))/n
  comparisons[bed]={'n':n,'hits':{m:int(sum(v[m])) for m in v},'delta_vs_hyde':delta,'wins_vs_hyde':wins,'losses_vs_hyde':losses,'per_query_full':v['full'],'per_query_hyde':v['hyde']}
  if sum(v['full'])<max(sum(v['plain']),sum(v['hyde']),sum(v['rerank'])):passed=False
 if len(comparisons)!=3 or not any(v['delta_vs_hyde']>=.05 and v['wins_vs_hyde']-v['losses_vs_hyde']>=3 for v in comparisons.values()):passed=False
 diag=diagnose(exclude_legacy=True);technical=diag.get('truncated',0)==0 and diag.get('thinking_content',0)==0 and diag.get('jcall_none',0)/max(1,diag.get('jcall_calls',0))<=.01 and diag.get('missing_items',0)/max(1,diag.get('expected_items',0))<=.01 and diag.get('batch_none_items',0)/max(1,diag.get('batch_return_items',0))<=.01
 result={'comparisons':comparisons,'diagnostics':diag,'performance_passed':passed,'technical_passed':technical,'expand_to_full':passed and technical,'interpretation':'screening for further full runs only, not a significance/novelty claim; final-vs-rerank comparison is reported separately'}
 dump('alignment-gate.json',result);return result['expand_to_full']
state={'started':time.strftime('%Y-%m-%d %H:%M:%S'),'deadline_unix':float(os.environ['NIGHT_DEADLINE_UNIX']),'completed':[],'failed':[],'current_job':None,'phase':'subset'}
try:
 # Check the model-independent original baseline before expensive model calls.
 for bed in ['inmind','lucid','implex']:
  run({'name':'subset-'+bed+'-plain','method':'plain','bed':bed+(',harm' if bed=='inmind' else '')})
 expected={'inmind':{'inmind_off':1,'inmind_dense':1,'harm':73},'lucid':{'lucid_test':42},'implex':{'implex_test':15}}
 check={}
 for bed,metrics in expected.items():
  r=json.loads((HERE/'results'/('subset-'+bed+'-plain.json')).read_text())['result'];check[bed]={k:{'actual':sum(r[k]),'author':n} for k,n in metrics.items()}
 dump('dense-baseline-alignment.json',check)
 if not all(v['actual']==v['author'] for bed in check.values() for v in bed.values()):raise RuntimeError('Model-independent dense baseline differs; stop before attributing differences to Qwen')
 # InMind source Round2 controls plus the actual final Round5 pipeline.
 for bed in ['inmind','lucid','implex']:
  for m in ['hyde','full','rerank']:
   run({'name':'subset-'+bed+'-'+m,'method':m,'bed':bed+(',harm' if bed=='inmind' else '')})
 for m in ['legacy_both','legacy_write','legacy_prior']:run({'name':'subset-inmind-'+m,'method':m,'bed':'inmind'})
 expand=gate()
 if expand:
  state['phase']='full';dump('status.json',state)
  with (HERE/'full-prepare.log').open('a') as f:code=subprocess.run([sys.executable,str(HERE/'prepare_full.py')],stdout=f,stderr=subprocess.STDOUT).returncode
  if code:raise RuntimeError('Full data audit failed; no full runs allowed')
  for m in ['plain','hyde','full','rerank']:run({'name':'full-inmind-'+m,'method':m,'bed':'inmind,harm','full':True})
  # Fixed folds preserve original per-query pool structure. Number derived without loading model modules.
  D=json.load(open(HERE/'snapshot/rsi-memory/bench_data/lucid/lucid.json'));D=[d for d in D if any(t.get('has_answer') for s in d['latent_sessions'] for t in s if t['role']=='user')];D.sort(key=lambda d:int(hashlib.md5(str(d['query_id']).encode()).hexdigest()[:8],16));G=collections.OrderedDict()
  for d in D:G.setdefault(d['query_id'],[]).append(d)
  folds=[];current=[]
  for g in G.values():
   if current and len(current)+len(g)>240:folds.append(current);current=[]
   current+=g
  if current:folds.append(current)
  # Match the author's global-score workload: 30 Implex records per fold, <=240 LUCid queries per fold.
  # Both benchmarks and all comparators progress together instead of running every baseline first.
  for i in range(max(52,len(folds))):
   if i<len(folds):
    for m in ['plain','hyde','full','rerank']:run({'name':'full-lucid-'+m+'-fold'+str(i),'method':m,'bed':'lucid','full':True,'fold':i})
   if i<52:
    for m in ['plain','hyde','full','rerank']:run({'name':'full-implex-'+m+'-fold'+str(i),'method':m,'bed':'implex','full':True,'fold':i})
  state['phase']='complete'
 else:state['phase']='subset_finished_full_not_triggered'
 saved_phase=state['phase'];state['phase']='source-answer-check';dump('status.json',state)
 for bed in ['inmind','implex']:
  for m in ['plain','hyde','full','rerank']:run({'name':'subset-answer-'+bed+'-'+m,'method':m,'bed':bed,'answer':True})
 state['phase']=saved_phase

except BudgetExpired:state['phase']='paused_at_budget'
except Exception:state['phase']='failed';state['error']=traceback.format_exc()
finally:
 state['ended']=time.strftime('%Y-%m-%d %H:%M:%S');state['diagnostics']=diagnose();dump('status.json',state);print(json.dumps(state),flush=True)
