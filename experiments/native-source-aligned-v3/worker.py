import os,sys,json,time,hashlib,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];BASE=ROOT/'experiments/original-qwen-format-v2';SRC=BASE/'snapshot'
os.environ.update(MEM_LM='openai',MEM_MODEL='qwen3-8b-b968826-bf16-nothink-vllm011',MEM_BASE_URL='http://127.0.0.1:8000/v1',MEM_API_KEY='local-only',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
os.environ.setdefault('NIGHT_DEADLINE_UNIX',os.environ.get('NATIVE_DEADLINE','inf'))
sys.path[:0]=[str(SRC/'rsi-memory'),str(SRC/'external-memory')]
import bench,rsi,ret,bridge,lm,answer
sys.path.insert(0,str(BASE));from instrument import install
install()
sys.path.insert(0,str(HERE))
from native import ingest,retrieve,T,B,close_memory
from transport import dump,event,budget,InvalidNative
METHODS=['plain','hyde','rerank','full','mem0','amem','rootmem']
def sha(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def prepare():
 out=HERE/'datasets.json'
 if out.exists():return json.loads(out.read_text())
 b=rsi.bed_inmind();all_tasks=b['testB']+b['stream']+b['gate']+b['testA']
 tasks=[{'q':t['q'],'targets':[t['target']],'fact':t['fact'],'pool':None,'id':t['id'],'heldout':i<61} for i,t in enumerate(all_tasks)]
 datasets={'inmind':{'lines':b['lines'],'tasks':tasks,'subset_n':61,'full_n':125,'protocol':'original 365-line shared dense pool; not per-query 241-line official-style pool'}}
 for name in ['implex','lucid']:
  bed=bench.bed_implex('test') if name=='implex' else bench.bed_lucid('test')
  facts=[q['trait'] for u in json.load(open(bench.IMPLEX))['test'] for q in u['qa']] if name=='implex' else [' / '.join(bed['lines'][j][:300] for j in t['targets'][:2]) for t in bed['tasks']]
  tasks=[]
  for i,(t,fact) in enumerate(zip(bed['tasks'],facts)):
   pool=bench._pool(t['pool'],len(bed['lines'])).tolist();tasks.append({'q':t['q'],'targets':t['targets'],'fact':fact,'pool':pool,'id':i,'heldout':True})
  datasets[name]={'lines':bed['lines'],'tasks':tasks,'subset_n':len(tasks),'protocol':'exact source test subset and same per-query candidate pool'}
 assert [datasets[x]['subset_n'] for x in ['inmind','lucid','implex']]==[61,240,81]
 dump(out,datasets);dump(HERE/'dataset-audit.json',{k:{'lines':len(v['lines']),'subset_questions':v['subset_n'],'total_questions':len(v['tasks']),'protocol':v['protocol'],'sha256':sha(v)} for k,v in datasets.items()});return datasets
def smoke(method):
 lines=['I am allergic to peanuts.','I live in Paris.','I have moved from Paris to Lyon.','My favorite activity is cycling.']
 m=ingest(method,HERE/'smoke'/method,lines)
 docs,meta=retrieve(method,m,'What food am I allergic to?',lines)
 assert 'peanut' in '\n'.join(docs).lower(),(method,'known-fact retrieval failed')
 if method=='rootmem':assert m['agent_skills'],'RootMem produced no skills for known durable facts'
 result={'method':method,'passed':True,'notes':4,'known_fact_retrieved':True,'update_input_included':True,'metadata':meta,'calls':T.calls,'native_failure_is_not_zero_score':True}
 dump(HERE/'checks'/(method+'.json'),result)
 close_memory(m,method)
def compare_contexts(bed,method,data):
 path=HERE/'contexts'/bed/(method+'.json');path.parent.mkdir(parents=True,exist_ok=True)
 rows=json.loads(path.read_text()) if path.exists() else {};lines=data['lines'];tasks=data['tasks']
 if method in ['none','oracle']:
  rows={str(i):{'units':[] if method=='none' else [t['fact']],'raw_hit6':None,'source_ids':[]} for i,t in enumerate(tasks)}
  dump(path,rows);return
 if method in ['plain','hyde','full','rerank']:
  design={'plain':'plain','hyde':'hyde','full':'hyde+prop','rerank':'hyde+prop'}[method]
  bench.W['prop']=8.;bridge.PV[0]=2;bench.CTRL[0]='nojoin' if method=='rerank' else 'wide' if method=='full' else '';bench.MIX[:]=['pick' if method in ['full','rerank'] else 'rrf',None]
  bridge.PV[0]=2
  scores=bench.scores(lines,[t['q'] for t in tasks],set(design.split('+')),[t['pool'] for t in tasks])
  import numpy as np
  for i,(t,s) in enumerate(zip(tasks,scores)):
   pool=bench._pool(t['pool'],len(lines));pick=pool[np.argsort(-s[pool])[:6]].tolist()
   rows[str(i)]={'units':[lines[j] for j in pick],'raw_hit6':any(j in t['targets'] for j in pick),'source_ids':pick}
  dump(path,rows);return
 # Build each candidate pool separately. No other user's notes can enter a pool.
 groups={}
 for i,t in enumerate(tasks):
  if str(i) in rows:continue
  pool=list(range(len(lines))) if t['pool'] is None else t['pool'];groups.setdefault(sha(pool),{'pool':pool,'indices':[]})['indices'].append(i)
 for group,g in groups.items():
  poollines=[lines[j] for j in g['pool']];state=HERE/'states'/bed/group/method;m=ingest(method,state,poollines)
  before=B.unchanged_digest(m,method) if method in ['mem0','amem'] else sha(m)
  for i in g['indices']:
   budget();docs,meta=retrieve(method,m,tasks[i]['q'],poollines)
   after=B.unchanged_digest(m,method) if method in ['mem0','amem'] else sha(m);assert before==after,'Queries mutated stored memories'
   rows[str(i)]={'units':docs,'raw_hit6':None,'metadata':meta,'frozen_state_sha256':before,'candidate_pool_sha256':sha(poollines)};dump(path,rows)
   event('retrieved',bed=bed,method=method,query=i,completed=len(rows),total=len(tasks))
  close_memory(m,method)
def evaluate(bed,method,n,data):
 contexts=json.loads((HERE/'contexts'/bed/(method+'.json')).read_text());out=HERE/'answers'/bed/(method+'.json');rows=json.loads(out.read_text()) if out.exists() else {}
 for i,t in enumerate(data['tasks'][:n]):
  budget()
  if str(i) in rows:continue
  c=contexts[str(i)];units=[str(x)[:400] for x in c['units'][:6]];notes='\n'.join('- '+x for x in units) or '(none)'
  response=lm.call(answer.ANS.format(notes=notes,q=t['q']))
  rows[str(i)]={'id':t['id'],'heldout':t['heldout'],'notes':notes,'answer':response,'applied':None,'raw_hit6':c['raw_hit6'],'judge_model':lm.MODEL,'same_model_judge':True}
  dump(out,rows);event('evaluated',bed=bed,method=method,completed=len(rows),requested=n)
 pending=[i for i in range(n) if rows[str(i)]['applied'] is None]
 for start in range(0,len(pending),10):
  budget();indices=pending[start:start+10]
  items='\n\n'.join(f"Item {j+1}\nHidden fact: {data['tasks'][i]['fact'][:400]}\nRequest: {data['tasks'][i]['q']}\nAnswer: {rows[str(i)]['answer']}" for j,i in enumerate(indices))
  judged=bridge.jcall(answer.JUDGE.format(items=items));assert [x['id'] for x in judged]==list(range(len(indices)))
  for i,result in zip(indices,judged):rows[str(i)]['applied']=result['v']=='yes'
  dump(out,rows);event('judged',bed=bed,method=method,done=sum(x['applied'] is not None for x in rows.values()),requested=n)
 selected=[rows[str(i)] for i in range(n)]
 dump(HERE/'results'/f'{bed}-{method}-{n}.json',{'bed':bed,'method':method,'n':n,'correct':sum(x['applied'] for x in selected),'metric':'same-backbone answer uses hidden fact; not raw retrieval hit','protocol':data['protocol'],'heldout_n':sum(x['heldout'] for x in selected),'raw_hit6':sum(x['raw_hit6'] for x in selected) if method in ['plain','hyde','rerank','full'] else None,'complete':True})
if __name__=='__main__':
 spec=json.loads(sys.argv[1]);start=time.time()
 import fcntl
 (HERE/'job-locks').mkdir(exist_ok=True)
 guard=(HERE/'job-locks'/('-'.join(str(spec.get(k,'')) for k in ['stage','bed','method'])+'.lock')).open('w')
 fcntl.flock(guard,fcntl.LOCK_EX)
 try:
  if spec['stage']=='smoke':smoke(spec['method'])
  else:
   data=prepare()[spec['bed']]
   if spec['stage']=='retrieve':compare_contexts(spec['bed'],spec['method'],data)
   else:evaluate(spec['bed'],spec['method'],spec['n'],data)
  event('job_complete',spec=spec,seconds=time.time()-start)
 except BaseException as e:
  event('job_failed',spec=spec,error=repr(e),traceback=traceback.format_exc());raise
