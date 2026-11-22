"""Pinned official Mem0/A-MEM algorithms; local backbone adaptation, not paper-score reproduction."""
from common import *
import logging, argparse, pickle, gc, copy, ast, fcntl
from types import SimpleNamespace
from datetime import datetime, timedelta
OUT=HERE/'baselines';OUT.mkdir(exist_ok=True)
T=Transport('baselines')
class Errors(logging.Handler):
 def __init__(self):super().__init__(logging.ERROR);self.errors=[]
 def emit(self,r):self.errors.append(r.getMessage())
errors=Errors();logging.getLogger().addHandler(errors)
FACT_SCHEMA={'type':'object','properties':{'facts':{'type':'array','items':{'type':'string'}}},'required':['facts']}
UPDATE_SCHEMA={'type':'object','properties':{'memory':{'type':'array','items':{'type':'object','properties':{'id':{'type':['string','integer']},'text':{'type':'string'},'event':{'enum':['ADD','UPDATE','DELETE','NONE']}},'required':['id','text','event']}}},'required':['memory']}
def mem0_new(path):
 from mem0 import Memory
 import importlib.metadata
 assert importlib.metadata.version('mem0ai')=='0.1.98', 'Use .venv-baselines/bin/python for the pinned paper-era Mem0'
 config={'version':'v1.1','history_db_path':str(path/'history.sqlite'),
  'vector_store':{'provider':'qdrant','config':{'collection_name':'inmind','path':str(path/'qdrant'),'embedding_model_dims':384,'on_disk':True}},
  'llm':{'provider':'openai','config':{'model':MODEL,'temperature':0,'max_tokens':2048,'openai_base_url':'http://127.0.0.1:8000/v1','api_key':'local-only'}},
  'embedder':{'provider':'huggingface','config':{'model':'BAAI/bge-small-en-v1.5','model_kwargs':{'device':'cpu','local_files_only':True}}}}
 m=Memory.from_config(config)
 def generate(messages,response_format=None,tools=None,tool_choice='auto',**kw):
  assert not tools,'Unexpected tool calling: local adapter does not support it'
  schema=FACT_SCHEMA if len(messages)==2 else UPDATE_SCHEMA
  return json.dumps(T.call(messages,schema=schema,json_output=True))
 m.llm.generate_response=generate
 return m
def amem_new():
 sys.path.insert(0,str(VENDOR/'A-mem-main'));import memory_layer as A
 if not hasattr(A,'_native_encoder'):A._native_encoder=A.SentenceTransformer
 def fixed_encoder(model_name):
  assert model_name=='BAAI/bge-small-en-v1.5', 'Unexpected embedding fallback '+model_name
  encoder=A._native_encoder(model_name,device='cpu',local_files_only=True)
  encoder.get_config_dict=lambda: {'model_name':model_name}
  return encoder
 A.SentenceTransformer=fixed_encoder
 class LocalController:
  def get_completion(self,prompt,response_format,temperature=0.7):
   schema=response_format.get('json_schema',{}).get('schema')
   return json.dumps(T.json(prompt,schema=schema))
 m=A.AgenticMemorySystem(model_name='BAAI/bge-small-en-v1.5',llm_backend='openai',llm_model=MODEL,api_key='local-only',refresh_index=False)
 m.llm_controller.llm=LocalController()
 tree=ast.parse((VENDOR/'A-mem-main/test_advanced.py').read_text())
 fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='generate_query_llm')
 module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));namespace={'json':json}
 exec(compile(module,'official_amem_query_preprocessor','exec'),namespace)
 m._query_keywords=lambda query: namespace['generate_query_llm'](SimpleNamespace(retriever_llm=m.llm_controller),query)
 return m,A
def unchanged_digest(m,name):
 if name=='amem':return hashlib.sha256(pickle.dumps(m.memories)).hexdigest()
 return hashlib.sha256(json.dumps(m.get_all(user_id='subject',limit=100000),sort_keys=True,default=str).encode()).hexdigest()
def retrieve(m,name,q):
 if name=='mem0':
  x=m.search(q,user_id='subject',limit=6);items=x['results'] if isinstance(x,dict) else x
  docs=[a['memory'] for a in items]
 else:
  keywords=m._query_keywords(q)
  raw=m.find_related_memories_raw(keywords,k=6)
  # Preserve native linked-neighbour expansion; enforce token budget at note boundaries.
  docs=re.split(r'\n(?=talk start time:)',raw)
 return budget_context(docs)
def judge(task,context,kind):
 system=(VENDOR/'InMind-main/evaluation/prompts/judge_target_recall.txt').read_text()
 payload={k:task[k] for k in ['user_message','explanation']};payload.update(context=context,query=task[kind])
 schema={'type':'object','properties':{'score':{'type':'integer','enum':[0,1]},'reason':{'type':'string'}},'required':['score','reason']}
 return T.call([{'role':'system','content':system},{'role':'user','content':json.dumps(payload)}],schema=schema,json_output=True)
def run(name,selected,smoke=False):
 allresults=[]
 for ti,task in enumerate(selected):
  path=OUT/(name+'-smoke-state' if smoke else name)/str(task['task_id'])
  if not smoke and path.exists() and not (path/'complete.json').exists():
   path.rename(path.with_name(path.name+'.interrupted-'+str(time.time_ns())))
  path.mkdir(parents=True,exist_ok=True)
  done=path/'complete.json'
  if done.exists():allresults.append(json.loads(done.read_text()));continue
  if name=='mem0':m=mem0_new(path);A=None
  else:m,A=amem_new()
  start=time.perf_counter();before_calls=T.calls;failures_start=dict(A.FAILURES) if A else {};chunks=[c for s in timeline(task) for c in pairs(s)]
  if smoke:chunks=[[{'role':'user','content':'I am allergic to peanuts.'}],[{'role':'user','content':'I live in Paris and enjoy cycling.'}]]
  for ci,c in enumerate(chunks):
   old_errors=len(errors.errors);old_fails=sum(v for k,v in A.FAILURES.items() if k!='invalid_link') if A else 0
   if name=='mem0':res=m.add(c,user_id='subject',infer=True)
   else:
    stamp=(datetime(2026,1,1)+timedelta(minutes=ci)).strftime('%Y%m%d%H%M')
    res=m.add_note(text(c),time=stamp)
   if len(errors.errors)!=old_errors:raise RuntimeError('Baseline logged error: '+str(errors.errors[old_errors:]))
   if A and sum(v for k,v in A.FAILURES.items() if k!='invalid_link')!=old_fails:raise RuntimeError('A-MEM absorbed algorithm failure: '+str(A.FAILURES))
   if ci%20==0:event('ingest',method=name,task=task['task_id'],chunk=ci+1,total=len(chunks),seconds=round(time.perf_counter()-start,1))
  if smoke and name=='amem':
   m.consolidate_memories()
   assert m.retriever.model.get_config_dict()['model_name']=='BAAI/bge-small-en-v1.5'
  freeze=unchanged_digest(m,name);records=[]
  if name=='amem':
   snapshot={key:vars(note) for key,note in m.memories.items()}
   if not smoke:
    dump(path/'amem-memory.json',snapshot)
    dump(path/'amem-retriever-corpus.json',m.retriever.corpus)
    np.save(path/'amem-embeddings.npy',m.retriever.embeddings)
   assert len(m.memories)==len(chunks), 'A-MEM ingestion lost a note'
   store_docs=[note.content for note in m.memories.values()]
  else:
   snapshot=m.get_all(user_id='subject',limit=100000)
   if not smoke:dump(path/'mem0-memory.json',snapshot)
   store_items=snapshot['results'] if isinstance(snapshot,dict) else snapshot
   store_docs=[item['memory'] for item in store_items]
  storage={'memory_count':len(store_docs),'literal_target':any(task['user_message'] in d for d in store_docs)}
  if not smoke and name=='mem0':
   pending=[];verdicts=[]
   for doc in store_docs:
    if pending and token_count('\n'.join(pending+[doc]))>6000:
     verdicts.append(judge(task,'\n'.join(pending),'query'));pending=[]
    pending.append(doc)
   if pending:verdicts.append(judge(task,'\n'.join(pending),'query'))
   storage.update(semantic_target=any(j['score']==1 for j in verdicts),judge_batches=verdicts)
  elif not smoke:storage['semantic_target']=storage['literal_target']
  for kind in ['query','naive_query']:
   q='What food am I allergic to?' if smoke else task[kind]
   ctx=retrieve(m,name,q)
   if smoke:
    assert 'peanut' in ctx.lower(),(name,'failed known-fact retrieval',ctx)
    j={'score':None,'reason':'smoke only'}
   else:j=judge(task,ctx,kind)
   assert unchanged_digest(m,name)==freeze,'query mutated frozen memory state'
   records.append({'kind':kind,'context':ctx,'context_tokens':token_count(ctx),'judge':j,'judge_model':MODEL,'judge_scope':'local same-backbone semantic target-recall diagnostic, not official gpt-5-mini leaderboard score'})
  row={'task_id':task['task_id'],'method':name,'chunks':len(chunks),'results':records,'seconds':time.perf_counter()-start,'calls':T.calls-before_calls,'state_query_isolation':True,'storage_diagnostic':storage,'native_failure_counts':{k:v-failures_start.get(k,0) for k,v in A.FAILURES.items()} if A else {}}
  if not smoke:dump(done,row)
  else:dump(OUT/(name+'-smoke.json'),row)
  allresults.append(row)
  if name=='mem0':m.vector_store.client.close()
  del m;gc.collect()
  event('baseline_task_done',method=name,task=task['task_id'],done=ti+1,total=len(selected),seconds=round(row['seconds'],1),calls=row['calls'])
 dump(OUT/(name+('-smoke-complete.json' if smoke else ('-all125.json' if len(selected)==125 else '-calibration-complete.json'))),allresults)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['mem0','amem'],required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--calibrate',action='store_true');a=ap.parse_args();tasks,_,_=load_data()
 with (OUT/(a.method+('-smoke' if a.smoke else '')+'.worker.lock')).open('w') as lock:
  event('method_lock_wait',method=a.method,pid=os.getpid())
  fcntl.flock(lock,fcntl.LOCK_EX)
  event('method_lock_acquired',method=a.method,pid=os.getpid())
  run(a.method,tasks[:1] if a.smoke or a.calibrate else tasks,smoke=a.smoke)
