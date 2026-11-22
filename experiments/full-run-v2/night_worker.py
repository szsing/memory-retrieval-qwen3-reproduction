"""Durable exhaustive job matrix, one source case per job; independent failures."""
from night_data import *
import argparse,traceback,subprocess
from night_budget import NightBudgetExpired,check_budget
RAW={'dense':['dense'],'hyde':['hyde'],'dense_pick':['dense_pick'],'properties':['both','write_only','query_only','shuffled','gated_pin','picker','fact_prior','gated_fact_prior']}
NATIVE=['mem0','amem','rootmem','rootmem_both']
def seed():
 init()
 with connect() as c:
  datasets=['inmind','lucid_s','implex','lucid_b'];cases={d:c.execute('select id,queries from cases where dataset=? order by rowid',(d,)).fetchall() for d in datasets}
  # First validate one full case on each dataset, then prioritize a complete InMind run.
  # This honors the earlier fallback: finish one full benchmark if three are too slow.
  order=[]
  for ds in datasets:
   if cases[ds]:order.append((ds,cases[ds][0]))
  for ds in datasets:
   order.extend((ds,item) for item in cases[ds][1:])
  for ds,(cid,qs) in order:
   if not json.loads(qs):continue
   for method in list(RAW)+NATIVE:c.execute("insert or ignore into jobs(dataset,id,method,status) values (?,?,?,'pending')",(ds,cid,method))
 event('night_queue_seeded')
def load_case(ds,cid):
 with connect() as c:
  h,qs=c.execute('select history,queries from cases where dataset=? and id=?',(ds,cid)).fetchone();history=json.loads(h);notes=[]
  for x in history:
   m,txt,ut=c.execute('select messages,text,user_text from notes where id=?',(x['note'],)).fetchone();notes.append({'id':x['note'],'messages':m,'text':txt,'user_text':ut,'session':x['session']})
 return notes,json.loads(qs)
def save_result(ds,cid,method,q,ctx,meta):
 value={'dataset':ds,'case_id':cid,'method':method,'qid':q['id'],'context':ctx,'context_tokens':token_count(ctx),'query':q['text'],'meta':meta,'gold':q['gold'],'heldout':q.get('heldout')}
 with connect() as c:c.execute('insert or ignore into results values (?,?,?,?,?)',(ds,cid,method,q['id'],json.dumps(value)))
def run_case(ds,cid,method):
 notes,queries=load_case(ds,cid)
 if method in RAW:
  from night_methods import orders
  oo=orders(notes,queries,method)
  for q,out in zip(queries,oo):
   for name,order in out.items():
    top=order[:6];ctx=budget_context([notes[i]['text'] for i in top]);sessions=set(notes[i]['session'] for i in top);gold=set(q['gold_sessions'])
    # Session hit and session recall are different metrics; raw note correctness is InMind-specific.
    meta={'top6_note_ids':[notes[i]['id'] for i in top],'top6_sessions':[notes[i]['session'] for i in top],'gold_session_hit6':int(bool(sessions&gold)) if gold else None,'gold_session_recall6':len(sessions&gold)/len(gold) if gold else None,'exact_raw_retrieval_metrics_applicable':True}
    if ds=='inmind':meta['target_note_hit6']=int(any(q['gold']['user_message'] in notes[i]['text'] and q['gold']['assistant_message'] in notes[i]['text'] for i in top))
    save_result(ds,cid,name,q,ctx,meta)
 else:
  from night_native import native_contexts
  for q,ctx,meta in native_contexts(ds,cid,method,notes,queries):save_result(ds,cid,method,q,ctx,meta)
 event('night_case_done',dataset=ds,id=cid,method=method,questions=len(queries))
def claim(method):
 with connect() as c:
  c.execute('begin immediate');r=c.execute("select dataset,id from jobs where method=? and status='pending' order by rowid limit 1",(method,)).fetchone()
  if r:c.execute("update jobs set status='running',updated=? where dataset=? and id=? and method=?",(time.time(),*r,method))
 return r

def worker(method):
 # No broad shared-state threads around native algorithms; native cases run in isolated processes.
 lockpath=HERE/f'night-{method}.lock'
 with lockpath.open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  # This lock guarantees no live owner when resetting jobs left by a killed prior worker.
  with connect() as c:c.execute("update jobs set status='pending' where method=? and status in ('running','paused_budget')",(method,))
  failures=0
  while True:
   r=claim(method)
   if not r:break
   ds,cid=r;start=time.time()
   try:
    check_budget()
    if method in NATIVE:
     log=HERE/'night-case-logs'/method/ds/(cid+'.log');log.parent.mkdir(parents=True,exist_ok=True)
     with log.open('a') as f:code=subprocess.run([sys.executable,'-u',__file__,'--case',ds,cid,'--method',method],stdout=f,stderr=subprocess.STDOUT).returncode
     if code==75:raise NightBudgetExpired('native case stopped at budget; completed LLM calls cached')
     if code:raise RuntimeError('native child exit '+str(code)+'; log='+str(log)+'; '+log.read_text()[-3500:])
    else:run_case(ds,cid,method)
    status='complete';err=None;failures=0
   except NightBudgetExpired as e:
    with connect() as c:c.execute("update jobs set status='paused_budget',error=?,updated=? where method=? and status in ('pending','running')",(e.reason,time.time(),method))
    event('night_method_paused',method=method,reason=e.reason);break
   except Exception:
    status='failed';failures+=1;err=traceback.format_exc();event('night_case_failed',dataset=ds,id=cid,method=method,error=err[-3000:])
   with connect() as c:c.execute('update jobs set status=?,error=?,updated=? where dataset=? and id=? and method=?',(status,err,time.time(),ds,cid,method))
   event('night_job_status',dataset=ds,id=cid,method=method,status=status,seconds=round(time.time()-start,1))
   if failures>=3:
    with connect() as c:c.execute("update jobs set status='blocked_after_failures' where method=? and status='pending'",(method,))
    event('night_method_stopped_after_three_failures',method=method);break
 event('night_worker_finished',method=method)
def evaluate_worker(shard):
 from night_eval import evaluate
 while True:
  try:check_budget()
  except NightBudgetExpired:return
  with connect() as c:rows=c.execute('select dataset,id,method,qid,value from results order by rowid').fetchall()
  work=[]
  for ds,cid,method,qid,raw in rows:
   if int(hashlib.sha256((ds+'|'+cid+'|'+method+'|'+qid).encode()).hexdigest(),16)%2!=shard:continue
   v=json.loads(raw)
   if 'evaluation' in v or 'evaluation_error' in v:continue
   work.append((ds,cid,method,qid,v))
  if not work:
   with connect() as c:left=c.execute("select count(*) from jobs where status in ('pending','running')").fetchone()[0]
   if left==0:break
   time.sleep(5);continue
  # Interleave datasets/methods, and rescan frequently for newly completed native cases.
  from collections import defaultdict,deque
  groups=defaultdict(deque)
  for item in work:groups[item[0],item[2]].append(item)
  priority=['both','dense','hyde','mem0','amem','rootmem','rootmem_both','dense_pick','write_only','query_only','shuffled','gated_pin','picker','fact_prior','gated_fact_prior']
  keys=sorted(groups,key=lambda k:(priority.index(k[1]),k[0]));work=[]
  while len(work)<32 and any(groups.values()):
   for key in keys:
    if groups[key]:work.append(groups[key].popleft())
    if len(work)>=32:break
  for ds,cid,method,qid,v in work:
   try:
    q={'id':qid,'text':v['query'],'gold':v['gold'],'heldout':v['heldout']}
    v['evaluation']=evaluate(ds,q,v['context'])
   except NightBudgetExpired:return
   except Exception:v['evaluation_error']=traceback.format_exc();event('night_eval_failed',dataset=ds,id=cid,method=method,qid=qid,error=v['evaluation_error'][-1500:])
   with connect() as c:c.execute('update results set value=? where dataset=? and id=? and method=? and qid=?',(json.dumps(v),ds,cid,method,qid))
   event('night_evaluated',dataset=ds,id=cid,method=method,qid=qid)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',action='store_true');p.add_argument('--method',choices=list(RAW)+NATIVE);p.add_argument('--case',nargs=2);p.add_argument('--eval',type=int,choices=[0,1]);a=p.parse_args()
 if a.seed:seed()
 elif a.case:run_case(*a.case,a.method)
 elif a.eval is not None:evaluate_worker(a.eval)
 else:worker(a.method)
