"""Isolated241 pools; freeze the identical240-line prefix, clone, append one fact."""
import os,sys,json,time,pickle,shutil,hashlib,traceback,copy
from pathlib import Path
from datetime import datetime,timedelta
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1];V3=ROOT/'experiments/native-source-aligned-v3'
sys.path.insert(0,str(V3))
import worker as W
import native as N
from transport import dump,budget,event,InvalidNative
METHOD=sys.argv[1]
D=json.loads((V3/'datasets.json').read_text())['inmind'];BG=D['lines'][:240];TASKS=D['tasks'];assert len(BG)==240 and len(TASKS)==125
BG_HASH=hashlib.sha256(json.dumps(BG,ensure_ascii=False).encode()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()
def close(m):
 if METHOD=='mem0':N.close_memory(m,'mem0');m.db.connection.close()
def canonical(m):
 if METHOD=='mem0':
  a=m.get_all(user_id='subject',limit=100000);a=a['results'] if isinstance(a,dict) else a;return sorted(x['memory'] for x in a)
 if METHOD=='amem':
  a=[{k:v for k,v in vars(x).items() if k not in ['id','last_accessed']} for x in m.memories.values()]
  return sorted(a,key=lambda x:(x['timestamp'],x['content']))
 return m
def save_result(rows,n):
 if all(str(i) in rows and rows[str(i)].get('applied') is not None for i in range(n)):
  selected=[rows[str(i)] for i in range(n)]
  dump(EXP/'results'/f'{METHOD}-{n}.json',{'method':METHOD,'n':n,'correct':sum(x['applied'] for x in selected),'metric':'same-Qwen hidden-fact application','raw_hit6':sum(x['raw_hit6'] for x in selected) if METHOD in ['plain','hyde','full','rerank'] else None,'pool_lines_per_question':241,'background_sha256':BG_HASH,'heldout_n':min(n,61),'all_complete':True})
def judge(rows,indices):
 if not indices:return
 items='\n\n'.join(f"Item {j+1}\nHidden fact: {TASKS[i]['fact'][:400]}\nRequest: {TASKS[i]['q']}\nAnswer: {rows[str(i)]['answer']}" for j,i in enumerate(indices))
 values=W.bridge.jcall(W.answer.JUDGE.format(items=items));assert [x['id'] for x in values]==list(range(len(indices)))
 for i,v in zip(indices,values):rows[str(i)]['applied']=v['v']=='yes'
 dump(EXP/'answers'/(METHOD+'.json'),rows)
def main():
 path=EXP/'answers'/(METHOD+'.json');rows=json.loads(path.read_text()) if path.exists() else {};pending=[];base=None;amem_blob=None;shared=None
 if METHOD in ['plain','hyde','full','rerank']:
  W.bench.W['prop']=8.;W.bridge.PV[0]=2;W.bench.CTRL[0]='wide' if METHOD=='full' else 'nojoin' if METHOD=='rerank' else '';W.bench.MIX[:]=['pick' if METHOD in ['full','rerank'] else 'rrf',None]
 if METHOD in ['mem0','amem','rootmem']:
  prefix=EXP/'prefix'/METHOD;shared=N.ingest(METHOD,prefix,BG)
  prefix_canonical=canonical(shared);dump(EXP/'prefix-audit'/(METHOD+'.json'),{'background_sha256':BG_HASH,'input_lines':240,'canonical_state_sha256':digest(prefix_canonical),'built_without_queries_or_target_facts':True})
  if METHOD=='mem0':close(shared);shared=None
  elif METHOD=='amem':
   amem_blob=pickle.dumps({'memories':shared.memories,'corpus':shared.retriever.corpus,'embeddings':shared.retriever.embeddings,'document_ids':shared.retriever.document_ids,'evo_cnt':shared.evo_cnt})
  else:base=copy.deepcopy(shared)
 for i,t in enumerate(TASKS):
  budget()
  if str(i) not in rows:
   lines=BG+[t['fact']];assert len(lines)==241
   if METHOD in ['plain','hyde','full','rerank']:
    W.bench.W['prop']=8.;W.bridge.PV[0]=2;W.bench.CTRL[0]='wide' if METHOD=='full' else 'nojoin' if METHOD=='rerank' else '';W.bench.MIX[:]=['pick' if METHOD in ['full','rerank'] else 'rrf',None]
    design={'plain':'plain','hyde':'hyde','full':'hyde+prop','rerank':'hyde+prop'}[METHOD]
    scores=W.bench.scores(lines,[t['q']],set(design.split('+')),[None])[0]
    import numpy as np
    picked=np.argsort(-scores)[:6].tolist();units=[lines[j] for j in picked];meta={'source_ids':picked};hit=240 in picked
   else:
    case=EXP/'states'/METHOD/str(i);case.parent.mkdir(parents=True,exist_ok=True)
    if METHOD=='mem0':
     if case.exists():shutil.rmtree(case)
     shutil.copytree(prefix,case)
     m=N.mem0(case);assert digest(canonical(m))==digest(prefix_canonical),'Prefix copy changed native memories'
     e=len(N.B.errors.errors);m.add([{'role':'user','content':t['fact']}],user_id='subject',infer=True)
     if len(N.B.errors.errors)!=e:raise InvalidNative('Mem0 native add logged an error')
    elif METHOD=='amem':
     m=shared;s=pickle.loads(amem_blob);m.memories=s['memories'];m.retriever.corpus=s['corpus'];m.retriever.embeddings=s['embeddings'];m.retriever.document_ids=s['document_ids'];m.evo_cnt=s['evo_cnt']
     import memory_layer as A
     before=dict(A.FAILURES);m.add_note(t['fact'],time=(datetime(2026,1,1)+timedelta(minutes=240)).strftime('%Y%m%d%H%M'))
     if any(v>before.get(k,0) for k,v in A.FAILURES.items() if k!='invalid_link'):raise InvalidNative('A-MEM native append failed')
    elif METHOD=='rootmem':
     sys.path.insert(0,str(N.VENDOR/'IMLogic-main/rootmem'))
     import build_rootmem as b
     case.mkdir(exist_ok=True);inp=case/'target.json';dump(inp,{'memory_points':[{'memory_content':t['fact'],'timestamp':'order_0000240'}]});target=b.extract_memories(inp)
     from types import SimpleNamespace
     class Client:
      def __init__(self):self.chat=SimpleNamespace(completions=self)
      def create(self,model,messages,response_format,temperature):
       value=N.T.call(messages,max_tokens=16384,schema=N.ROOT_SCHEMA,json_output=True)
       return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))],usage=None)
     if target:m,_=b.update_matrix_with_llm(Client(),copy.deepcopy(base),target,5,case/'build.log')
     else:m=copy.deepcopy(base)
    else:
     import numpy as np
     design={'plain':'plain','hyde':'hyde','full':'hyde+prop','rerank':'hyde+prop'}[METHOD]
     scores=W.bench.scores(lines,[t['q']],set(design.split('+')),[None])[0];ids=np.argsort(-scores)[:6].tolist();units=[lines[j] for j in ids];meta={'source_ids':ids};hit=240 in ids
    if METHOD in ['mem0','amem','rootmem']:
     state_digest=digest(canonical(m))
     if i==0 and not (EXP/'prefix-audit'/(METHOD+'-fresh-equivalence.json')).exists():
      fresh=N.ingest(METHOD,EXP/'golden-fresh'/METHOD,lines);equivalent=digest(canonical(fresh))==state_digest
      dump(EXP/'prefix-audit'/(METHOD+'-fresh-equivalence.json'),{'fresh241_equals_cloned240_plus_target':equivalent,'opaque_uuid_and_wall_clock_last_access_excluded':True,'fresh_state_sha256':digest(canonical(fresh)),'cloned_state_sha256':state_digest})
      close(fresh)
      if not equivalent:raise InvalidNative('Prefix cloning equivalence failed; do not expand this method')
     if METHOD=='rootmem':dump(case/'matrix.json',m)
     units,meta=N.retrieve(METHOD,m,t['q'],lines);assert digest(canonical(m))==state_digest,'Query changed native memory'
     close(m);hit=None
   notes='\n'.join('- '+str(x)[:400] for x in units[:6]) or '(none)';response=W.lm.call(W.answer.ANS.format(notes=notes,q=t['q']))
   rows[str(i)]={'id':t['id'],'heldout':i<61,'pool_sha256':digest(lines),'pool_lines':241,'target_fact_count':1,'notes':notes,'answer':response,'applied':None,'raw_hit6':hit,'metadata':meta};dump(path,rows)
  if rows[str(i)]['applied'] is None:pending.append(i)
  if len(pending)==10:judge(rows,pending);pending=[]
  if i in [60,124]:judge(rows,pending);pending=[];save_result(rows,i+1)
  dump(EXP/'progress'/(METHOD+'.json'),{'method':METHOD,'answered':len(rows),'scored':sum(x.get('applied') is not None for x in rows.values()),'current_i':i,'total':125,'updated':time.strftime('%H:%M:%S')})
 W.ret.save();event('strict241_method_complete',method=METHOD)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  dump(EXP/'errors'/(METHOD+'.json'),{'method':METHOD,'error':repr(e),'traceback':traceback.format_exc(),'time':time.strftime('%Y-%m-%d %H:%M:%S')});raise
