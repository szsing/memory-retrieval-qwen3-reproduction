"""Official native memory algorithms, unchanged construction/evolution prompts."""
from night_data import *
import run_baselines as B
from night_budget import NightTransport
B.T=NightTransport('night-native-baselines')
import pickle,gc,copy
from datetime import datetime,timedelta
from types import SimpleNamespace
from night_budget import NightTransport
T=NightTransport('night-rootmem')
def native_contexts(dataset,cid,method,notes,queries):
 path=HERE/'night-state'/method/dataset/cid;path.mkdir(parents=True,exist_ok=True)
 snapshot=path/'ingestion-complete.json'
 if method.startswith('rootmem'):
  yield from root_contexts(path,method,notes,queries);return
 if snapshot.exists():
  if method=='amem':
   m,A=B.amem_new();state=pickle.loads((path/'state.pkl').read_bytes())
   m.memories=state['memories'];m.retriever.corpus=state['corpus'];m.retriever.embeddings=state['embeddings'];m.retriever.document_ids=state['document_ids'];m.evo_cnt=state['evo_cnt']
  else:m=B.mem0_new(path);A=None
 else:
  # An interrupted add may have changed native storage. Rebuild with exact cached calls.
  if (path/'dirty').exists():
   backup=path.with_name(path.name+'.interrupted-'+str(time.time_ns()));path.rename(backup);path.mkdir()
  (path/'dirty').write_text('Do not resume a partially committed native add; replay from validated LLM cache.\n')
  if method=='mem0':m=B.mem0_new(path);A=None
  else:m,A=B.amem_new()
  start=time.perf_counter()
  for i,n in enumerate(notes):
   e=len(B.errors.errors);f=sum(v for k,v in A.FAILURES.items() if k!='invalid_link') if A else 0
   if method=='mem0':m.add(json.loads(n['messages']),user_id='subject',infer=True)
   else:m.add_note(n['text'],time=(datetime(2026,1,1)+timedelta(minutes=i)).strftime('%Y%m%d%H%M'))
   if len(B.errors.errors)!=e:raise RuntimeError('Native error: '+str(B.errors.errors[e:]))
   if A and sum(v for k,v in A.FAILURES.items() if k!='invalid_link')!=f:raise RuntimeError('Native A-MEM failure: '+str(A.FAILURES))
   if i%20==0:event('night_native_ingest',dataset=dataset,id=cid,method=method,done=i+1,total=len(notes))
  if method=='amem':
   assert len(m.memories)==len(notes)
   state={'memories':m.memories,'corpus':m.retriever.corpus,'embeddings':m.retriever.embeddings,'document_ids':m.retriever.document_ids,'evo_cnt':m.evo_cnt}
   (path/'state.pkl').write_bytes(pickle.dumps(state));dump(path/'memories.json',{k:vars(n) for k,n in m.memories.items()})
  else:dump(path/'memories.json',m.get_all(user_id='subject',limit=100000))
  dump(snapshot,{'chunks':len(notes),'model':MODEL,'logical_clock_not_event_dates':True,'seconds':time.perf_counter()-start,'uncached_llm_calls':B.T.calls,'cache_hits':B.T.cached,'native_failure_counts':dict(A.FAILURES) if A else {}})
  (path/'dirty').unlink()
 frozen=B.unchanged_digest(m,method)
 for q in queries:
  ctx=B.retrieve(m,method,q['text']);assert B.unchanged_digest(m,method)==frozen
  yield q,ctx,{'frozen_state_sha256':frozen,'exact_raw_retrieval_metrics_applicable':False,'ingestion':json.loads(snapshot.read_text())}
 if method=='mem0':m.vector_store.client.close()
 del m;gc.collect()
def root_contexts(path,method,notes,queries):
 sys.path.insert(0,str(VENDOR/'IMLogic-main/rootmem'));import build_rootmem as b,use_rootmem as u
 from night_methods import vec,ranked
 b.LLM_MODEL_NAME=MODEL;b.TEMPERATURE=0;u.ROUTER_MODEL_NAME=MODEL;u.ROUTER_TEMPERATURE=0;u.MAX_TOTAL_ITEMS=30
 class Client:
  def __init__(self):self.chat=SimpleNamespace(completions=self)
  def create(self,model,messages,response_format,temperature):
   v=T.call(messages,max_tokens=8192,response_format=response_format,json_output=True)
   # Source stats are unused; usage not invented or reported as actual zero.
   return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(v)))],usage=None)
 client=Client();inputfile=path/'input-memory.json'
 if not inputfile.exists():dump(inputfile,{'memory_points':[{'memory_content':n['user_text'],'timestamp':f'order_{i:07d}'} for i,n in enumerate(notes) if n['user_text'].strip()]})
 memories=b.extract_memories(inputfile);checkpoint=path/'matrix-checkpoint.json'
 state=json.loads(checkpoint.read_text()) if checkpoint.exists() else {'next':0,'matrix':{'agent_skills':[]}}
 # Native numeric reference BATCH_SIZE=60. No dropping older memories.
 offset=state['next'];batch_index=state.get('batch_index',0)
 while offset<len(memories):
  end=min(offset+60,len(memories));batch_index+=1
  while end>offset:
   prompt=b.build_user_prompt(state['matrix'],memories[offset:end],batch_index)
   if token_count(b.SYSTEM_PROMPT)+token_count(prompt)+8192+512<=32768:break
   end-=1
  if end==offset:raise ValueError('One whole RootMem input memory exceeds local context; no truncation allowed')
  matrix,_=b.update_matrix_with_llm(client,state['matrix'],memories[offset:end],batch_index,path/'build.log')
  state={'next':end,'batch_index':batch_index,'matrix':matrix};dump(checkpoint,state);offset=end
  event('rootmem_batch',id=path.name,done=offset,total=len(memories))
 skills=state['matrix']['agent_skills'];E=vec([n['text'] for n in notes]);from night_methods import match,props
 # RootMem+properties uses the already frozen both-side RRF method as raw retrieval.
 from night_methods import orders
 enhanced=orders(notes,queries,'properties') if method=='rootmem_both' else None
 for i,q in enumerate(queries):
  order=enhanced[i]['both'] if enhanced else ranked(E@vec([q['text']])[0])
  domains,meta=u.get_active_skills(q['text'],skills,client)
  gs,ctx,active,budget=u.build_mode_context('DYNAMIC_MATRIX_RAG',skills,domains,[notes[j]['text'] for j in order[:6]])
  # Whole active skill units then whole raw notes within a common 4096-token budget.
  units=[json.dumps(s,ensure_ascii=False) for s in active]+[notes[j]['text'] for j in order[:min(6,budget)]]
  context=budget_context(units)
  yield q,context,{'router':meta,'activated_domains':domains,'matrix_sha256':sha(checkpoint),'exact_raw_retrieval_metrics_applicable':False,'adaptation':'native RootMem construction/router, raw user-memory input, local BGE raw retrieval, common whole-unit token budget/reader'}
