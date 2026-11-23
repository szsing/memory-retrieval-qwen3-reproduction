import os,sys,json,time,hashlib,shutil,traceback
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[1];deadline=time.time()+7200
os.environ.update(MEM_PROJECT_ROOT=str(R),MEM_LM='openai',MEM_MODEL='qwen3-8b-b968826-bf16-nothink-vllm011',MEM_BASE_URL='http://127.0.0.1:8000/v1',MEM_API_KEY='local-only',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NIGHT_DEADLINE_UNIX=str(deadline),RUN_JOB='haiku-swap-20261123')
BASE=R/'experiments/original-qwen-format-v2';SRC=BASE/'snapshot';sys.path[:0]=[str(SRC/'rsi-memory'),str(SRC/'external-memory'),str(BASE)]
import numpy as np
import bench,rsi,bridge,ret,lm,methods as M
from instrument import install
install()
def dump(n,d):
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def rr(x):
 r=np.empty(len(x));r[np.argsort(-x)]=np.arange(len(x));return 1/(60+r)
def main():
 H=json.loads((O/'haiku-cache.json').read_text());bed=rsi.bed_inmind();lines=bed['lines'];tasks=bed['testB'];qs=[t['q'] for t in tasks];assert len(lines)==365 and len(qs)==61
 assert [x['text'] for x in H['memory_lines']]==lines;assert [x['q'] for x in H['requests']]==qs
 assert all(h['fact_line']==t['target'] for h,t in zip(H['requests'],tasks))
 for x in H['memory_lines']+H['requests']:assert isinstance(x['haiku_properties'],list) and all(isinstance(v,str) for v in x['haiku_properties'])
 cache=O/'embedding-cache';cache.mkdir(exist_ok=True);src=Path(ret._path())
 if src.exists() and not (cache/'emb.npz').exists():shutil.copy2(src,cache/'emb.npz')
 ret.CDIR=str(cache);ret.save=lambda:None
 bridge.PV[0]=2;bench.W['prop']=8.;bench.MIX[:]=['pick',None];bench.CTRL[0]='wide'
 original_lp=bridge.props_lines;original_rp=bridge.props_reqs;original_gate=bridge.gate_reqs;original_hyde=M.gen_hyde;original_bonus=bridge.PropIndex.bonus;original_picker=bridge.pick_notes
 st=rsi.Store(dict(lines=lines,nb=240));plain=st.scores(qs);plain_pick=[np.argsort(-s)[:6].tolist() for s in plain];assert sum(t['target'] in p for t,p in zip(tasks,plain_pick))==1,'plain baseline mismatch'
 config={'started':time.strftime('%F %T'),'deadline_unix':deadline,'n':61,'pool':'original shared365, full-pool ranking; off-style score logged separately','model':lm.MODEL,'temperature':0,'max_tokens':2048,'thinking':False,'prompt_version':2,'weight':8,'picker':'wide dense12+joined6, select6','frozen_components':['Qwen61-batched HyDE','Qwen61-batched gate','embedding checkpoint','prompt templates','weight/picker settings'],'groups':['QQ','HQ','QH','HH','H_raw_query','HH_presence','HH_shuffled'],'selection':'all first61 unseen-domain original testB; no tuning','cache_raw_sha256':hashlib.sha256((O/'haiku-cache.json').read_bytes()).hexdigest(),'plain_hit6':1,'labels_not_used_as_features':['fact_line','kind']};dump('protocol.json',config)
 dump('status.json',{'phase':'generating_or_recovering_Qwen_features','updated':time.strftime('%F %T')})
 f=O/'qwen-features.json'
 if not f.exists():
  before=dict(lm.STATS);F={'memory_properties':original_lp(lines),'request_properties':original_rp(qs),'hyde':original_hyde(qs),'gate':original_gate(qs),'line_sha256':[sha(x) for x in lines],'query_sha256':[sha(x) for x in qs],'generation_stats_before':before,'generation_stats_after':dict(lm.STATS)};dump('qwen-features.json',F)
 F=json.loads(f.read_text());assert F['line_sha256']==[sha(x) for x in lines] and F['query_sha256']==[sha(x) for x in qs]
 HL=[x['haiku_properties'] for x in H['memory_lines']];HQ=[x['haiku_properties'] for x in H['requests']];lookup={q:i for i,q in enumerate(qs)}
 M.gen_hyde=lambda q:[F['hyde'][lookup[x]] for x in q];bridge.gate_reqs=lambda q:[F['gate'][lookup[x]] for x in q]
 dense=st.scores(qs,[[x] if x else [] for x in F['hyde']]);summary={}
 for name in config['groups']:
  if time.time()>deadline:raise TimeoutError('experiment window expired')
  done=O/'groups'/(name+'.json')
  if done.exists():summary[name]=json.loads(done.read_text())['summary'];continue
  LP=F['memory_properties'] if name in ['QQ','QH'] else HL
  RP=F['request_properties'] if name in ['QQ','HQ'] else HQ
  if name=='H_raw_query':RP=[[q] for q in qs]
  if name=='HH_shuffled':RP=RP[len(RP)//2:]+RP[:len(RP)//2]
  bridge.props_lines=lambda ls:LP if ls==lines else (_ for _ in ()).throw(ValueError('memory order changed'))
  bridge.props_reqs=lambda q:[RP[lookup[x]] for x in q]
  captured={};candidate={}
  def bonus(self,rp):
   B=original_bonus(self,rp)
   if name=='HH_presence':
    has=np.zeros(self.n);has[self.own]=1
    B=[(has if p else 0.)+1e-3*s for p,s in zip(rp,dense)]
   captured['B']=B;return B
  bridge.PropIndex.bonus=bonus
  def pick(q,docs):
   for query,notes in zip(q,docs):
    i=lookup[query];s=dense[i];b=captured['B'][i];z=rr(s)+8*rr(b);z[np.argsort(-s)[:1]]+=100
    c=list(dict.fromkeys(np.argsort(-s)[:12].tolist()+np.argsort(-z)[:6].tolist()));assert [lines[j] for j in c]==notes,'candidate reconstruction mismatch'
    candidate[i]={'source_ids':c,'index_target_present':tasks[i]['target'] in c,'target_text_present':lines[tasks[i]['target']] in notes}
   picks=original_picker(q,docs)
   for query,p in zip(q,picks):candidate[lookup[query]]['picker_local_indices']=p
   return picks
  bridge.pick_notes=pick;os.environ['RUN_JOB']='haiku-swap-'+name
  dump('status.json',{'phase':'running','group':name,'complete':summary,'updated':time.strftime('%F %T')})
  scores=bench.scores(lines,qs,{'hyde','prop'},[None]*61);records=[]
  for i,(t,s) in enumerate(zip(tasks,scores)):
   ids=np.argsort(-s)[:6].tolist();b=captured['B'][i]
   records.append({'i':i,'task_id':t['id'],'query_sha256':sha(qs[i]),'target':t['target'],'selected_ids':ids,'hit6':t['target'] in ids,'text_hit6':any(lines[j]==lines[t['target']] for j in ids),'off_style_hit6':int((s[:240]>s[t['target']]).sum())<6,'attribute_target_rank':int((b>b[t['target']]).sum())+1,'candidate':candidate.get(i),'gate':F['gate'][i]})
  result={'hit6':sum(r['hit6'] for r in records),'text_hit6':sum(r['text_hit6'] for r in records),'off_style_hit6':sum(r['off_style_hit6'] for r in records),'candidate_covered':sum(bool(r['candidate'] and r['candidate']['index_target_present']) for r in records),'n':61,'lm_stats':dict(lm.STATS)}
  dump('groups/'+name+'.json',{'summary':result,'records':records});summary[name]=result;dump('summary.json',summary)
 dump('status.json',{'phase':'complete','complete':summary,'updated':time.strftime('%F %T')})
if __name__=='__main__':
 try:main()
 except BaseException as e:dump('error.json',{'error':repr(e),'traceback':traceback.format_exc(),'time':time.strftime('%F %T')});raise
