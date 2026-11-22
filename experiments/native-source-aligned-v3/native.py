"""Pinned native algorithms; only backend and explicitly documented input/output adapters."""
import os,sys,json,ast,re,pickle,hashlib,logging
from pathlib import Path
from datetime import datetime,timedelta
from types import SimpleNamespace
from transport import *
ROOT=HERE.parents[1];OLD=ROOT/'experiments/full-run-v2';VENDOR=OLD/'vendor'
sys.path.insert(0,str(OLD));os.environ['MEM_PROJECT_ROOT']=str(ROOT)
import run_baselines as B
T=Transport();B.T=T
ROOT_SCHEMA={'type':'object','properties':{'agent_skills':{'type':'array','items':{'type':'object','properties':{k:{'type':'string'} for k in ['Skill_Domain','Execution_Rules','Logic_Aware_Decision_Region']},'required':['Skill_Domain','Execution_Rules','Logic_Aware_Decision_Region'],'additionalProperties':False}}},'required':['agent_skills'],'additionalProperties':False}
def mem0(path):
 import mem0.memory.main as mem0_main
 mem0_main.mem0_dir=str(path/'sdk-internal')
 m=B.mem0_new(path)
 def generate(messages,response_format=None,tools=None,**kw):
  assert not tools
  if len(messages)==2:schema=B.FACT_SCHEMA
  else:
   # The native prompt exposes temporary IDs. Constrain references to those IDs only.
   text=messages[0]['content'];part=text.split('Below is the current content of my memory',1)[1]
   old=ast.literal_eval(re.search(r'```\s*(.*?)\s*```',part,re.S)[1]);ids=[str(x['id']) for x in old]
   props={'id':{'type':'string'},'text':{'type':'string'},'event':{'const':'ADD'},'old_memory':{'type':'string'}}
   variants=[{'type':'object','properties':props,'required':['event','text'],'additionalProperties':False}]
   if ids:
    props2={**props,'id':{'type':'string','enum':ids},'event':{'enum':['UPDATE','DELETE','NONE']}}
    variants.append({'type':'object','properties':props2,'required':['event','id','text'],'additionalProperties':False})
   schema={'type':'object','properties':{'memory':{'type':'array','items':{'anyOf':variants}}},'required':['memory'],'additionalProperties':False}
  return json.dumps(T.call(messages,schema=schema,json_output=True))
 m.llm.generate_response=generate;return m
def amem():return B.amem_new()
def ingest(method,path,lines):
 path.mkdir(parents=True,exist_ok=True);signature=hashlib.sha256(json.dumps(lines).encode()).hexdigest()
 if (path/'ready.json').exists():
  assert json.loads((path/'ready.json').read_text())['input_sha256']==signature
  if method=='amem':
   m,A=amem();s=pickle.loads((path/'state.pkl').read_bytes())
   m.memories=s['memories'];m.retriever.corpus=s['corpus'];m.retriever.embeddings=s['embeddings'];m.retriever.document_ids=s['document_ids'];m.evo_cnt=s['evo_cnt'];return m
  if method=='mem0':return mem0(path)
  return json.loads((path/'matrix.json').read_text())
 if (path/'dirty').exists() and method=='mem0':
  import time
  moved=path.with_name(path.name+'.interrupted-'+str(time.time_ns()));path.rename(moved);path.mkdir()
 (path/'dirty').touch()
 if method=='rootmem':
  sys.path.insert(0,str(VENDOR/'IMLogic-main/rootmem'));import build_rootmem as b
  b.LLM_MODEL_NAME=MODEL;b.TEMPERATURE=0
  class Client:
   def __init__(self):self.chat=SimpleNamespace(completions=self)
   def create(self,model,messages,response_format,temperature):
    v=T.call(messages,max_tokens=16384,schema=ROOT_SCHEMA,json_output=True)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(v)))],usage=None)
  inp=path/'input.json';dump(inp,{'memory_points':[{'memory_content':s,'timestamp':f'order_{i:07d}'} for i,s in enumerate(lines)]})
  memories=b.extract_memories(inp);cp=path/'checkpoint.json';state=json.loads(cp.read_text()) if cp.exists() else {'next':0,'matrix':{'agent_skills':[]}}
  while state['next']<len(memories):
   budget();offset=state['next'];end=min(offset+60,len(memories));batch=offset//60+1
   # Whole memories only; never truncate source text to fit the local context window.
   while end>offset and B.token_count(b.SYSTEM_PROMPT+b.build_user_prompt(state['matrix'],memories[offset:end],batch))+16384+512>32768:end-=1
   if end==offset:raise InvalidNative('RootMem input exceeds context')
   matrix,_=b.update_matrix_with_llm(Client(),state['matrix'],memories[offset:end],batch,path/'build.log')
   state={'next':end,'matrix':matrix};dump(cp,state);event('ingest',method=method,done=end,total=len(memories))
  m=state['matrix'];dump(path/'matrix.json',m)
 else:
  if method=='mem0':m=mem0(path);A=None
  else:m,A=amem()
  for i,s in enumerate(lines):
   budget();e=len(B.errors.errors);fail=dict(A.FAILURES) if A else {}
   if method=='mem0':m.add([{'role':'user','content':s}],user_id='subject',infer=True)
   else:m.add_note(s,time=(datetime(2026,1,1)+timedelta(minutes=i)).strftime('%Y%m%d%H%M'))
   if len(B.errors.errors)!=e:raise InvalidNative('Native logged errors: '+str(B.errors.errors[e:]))
   if A and any(A.FAILURES[k]>fail.get(k,0) for k in A.FAILURES if k!='invalid_link'):raise InvalidNative('A-MEM absorbed failures: '+str(A.FAILURES))
   if A and A.FAILURES['invalid_link']>fail.get('invalid_link',0):event('native_rejected_link',method=method,count=A.FAILURES['invalid_link']-fail.get('invalid_link',0),policy='native invalid-link guard retained; no invalid edge inserted')
   if i%10==0:event('ingest',method=method,done=i+1,total=len(lines))
  if method=='amem':
   assert len(m.memories)==len(lines)
   state={'memories':m.memories,'corpus':m.retriever.corpus,'embeddings':m.retriever.embeddings,'document_ids':m.retriever.document_ids,'evo_cnt':m.evo_cnt}
   (path/'state.pkl').write_bytes(pickle.dumps(state));dump(path/'memories.json',{k:vars(v) for k,v in m.memories.items()})
  else:dump(path/'memories.json',m.get_all(user_id='subject',limit=100000))
 dump(path/'ready.json',{'input_sha256':signature,'n':len(lines),'method':method,'native_ingestion_complete':True});(path/'dirty').unlink();return m
def retrieve(method,m,q,lines):
 if method=='mem0':
  x=m.search(q,user_id='subject',limit=6);x=x['results'] if isinstance(x,dict) else x;return [a['memory'] for a in x],{}
 if method=='amem':
  before=B.unchanged_digest(m,'amem');fail=dict(__import__('memory_layer').FAILURES)
  k=m._query_keywords(q);out=m.find_related_memories_raw(k,k=6)
  assert B.unchanged_digest(m,'amem')==before
  if any(v>fail.get(k,0) for k,v in __import__('memory_layer').FAILURES.items() if k!='invalid_link'):raise InvalidNative('A-MEM query absorbed failure')
  return re.split(r'\n(?=talk start time:)',out),{'frozen_state_unchanged':True}
 sys.path.insert(0,str(VENDOR/'IMLogic-main/rootmem'));import use_rootmem as u
 u.ROUTER_MODEL_NAME=MODEL;u.ROUTER_TEMPERATURE=0;u.MAX_TOTAL_ITEMS=30;skills=m['agent_skills']
 schema={'type':'object','properties':{'activated_domains':{'type':'array','items':{'type':'string','enum':[s['Skill_Domain'] for s in skills]}}},'required':['activated_domains'],'additionalProperties':False} if skills else None
 class Client:
  def __init__(self):self.chat=SimpleNamespace(completions=self)
  def create(self,model,messages,response_format,temperature):
   v=T.call(messages,schema=schema,response_format=response_format,json_output=True)
   return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(v)))])
 domains,meta=u.get_active_skills(q,skills,Client())
 if meta.get('fallback_to_all'):raise InvalidNative('RootMem router failure: '+str(meta))
 import ret,numpy as np
 scores=ret.embed(lines)@ret.embed([q],query=True)[0];raw=[lines[i] for i in np.argsort(-scores)[:6]]
 _,_,active,rag_budget=u.build_mode_context('DYNAMIC_MATRIX_RAG',skills,domains,raw)
 # Equal answer budget: max six memory units, max400 chars each, as original answer.py.
 return [json.dumps(s,ensure_ascii=False) for s in active]+raw[:min(6,rag_budget)],{'router':meta,'activated_domains':domains,'external_retrieval_replaced_by':'same BGE-small raw search'}

def close_memory(m,method):
 if method=='mem0':
  m.vector_store.client.close()
  m._telemetry_vector_store.client.close()
