"""Run unchanged upstream functions with a local backend and observational diagnostics."""
import os,sys,json,time,hashlib,traceback,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SRC=HERE/'snapshot';R=SRC/'rsi-memory'
os.environ.update(MEM_LM='openai',MEM_MODEL='qwen3-8b-b968826-bf16-nothink-vllm011',MEM_BASE_URL='http://127.0.0.1:8000/v1',MEM_API_KEY='local-only',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
sys.path[:0]=[str(R),str(SRC/'external-memory')]
from instrument import install,budget,log
lm=install()
import bench,bridge,rsi,ret
METHODS={
 'plain':dict(design='plain',mix='rrf',w=2.,pv=1,ctrl=''),
 'hyde':dict(design='hyde',mix='rrf',w=2.,pv=1,ctrl=''),
 'full':dict(design='hyde+prop',mix='pick',w=8.,pv=2,ctrl='wide'),
 'rerank':dict(design='hyde+prop',mix='pick',w=8.,pv=2,ctrl='nojoin'),
 'legacy_both':dict(design='prop',mix='rrf',w=2.,pv=1,ctrl='',side='both'),
 'legacy_write':dict(design='prop',mix='rrf',w=2.,pv=1,ctrl='',side='line'),
 'legacy_prior':dict(design='prop',mix='rrf',w=2.,pv=1,ctrl='null',side='both')}
def dump(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
def configure(c):
 bench.W['prop']=c['w'];bridge.PV[0]=c['pv'];bench.CTRL[0]=c['ctrl'];bench.MIX[:]=[c['mix'],None]
 bridge.MIX=c['mix'];bridge.CTRL=c['ctrl'];bridge.SIDE=c.get('side','both');bridge.LLMSEL=False

def run_legacy(c):
 b=rsi.bed_inmind();st=rsi.Store(b);idx=bridge.PropIndex(bridge.props_lines(st.lines));ans={}
 for name in ['testA','testB']:
  dense,off=bridge.score_set(st,b[name],idx,c['w'],False)
  ans[name]={'dense':dense,'official':off,'task_ids':[t['id'] for t in b[name]]}
 dense,off=bridge.score_set(st,b['testA']+b['testB'],idx,c['w'],False,naive=True)
 ans['direct']={'dense':dense,'official':off};return ans

def full_data(bed,index):
 if bed=='inmind':
  original=rsi.bed_inmind();tasks=original['stream']+original['gate']+original['testA']+original['testB']
  # Expanded task coverage only: original 365-line store and original scoring kept.
  def expanded():
   ind=[dict(q=t['q'],targets=[t['target']],pool=None,off=True) for t in tasks]
   nav=[dict(q=t['naive'],targets=[t['target']],pool=None,off=True) for t in tasks]
   return original['lines'],original['nb'],ind,nav
  bench.bed_inmind_parts=expanded
 elif bed=='implex':
  path=HERE/'data'/('implex-fold-'+str(index)+'.json')
  if not path.exists():
   users=json.load(open(HERE/'data/implexconv_full_same_protocol.json'))['test'][index*30:(index+1)*30];dump(path,{'val':users,'test':users})
  bench.IMPLEX=str(path)
  return {'fold':index,'row_start':index*30,'rows':min(30,1550-index*30)}
 elif bed=='lucid':
  D=json.load(open(bench.LUCID));D=[d for d in D if any(t.get('has_answer') for s in d['latent_sessions'] for t in s if t['role']=='user')];D.sort(key=lambda d:rsi.h(d['query_id']))
  # Full-query extension in fixed query-id-group folds. No query shares its own evidence session with the donor pool.
  grouped=collections.OrderedDict()
  for d in D:grouped.setdefault(d['query_id'],[]).append(d)
  folds=[];current=[]
  for g in grouped.values():
   if current and len(current)+len(g)>240:folds.append(current);current=[]
   current+=g
  if current:folds.append(current)
  items=folds[index];qids={d['query_id'] for d in items};avoid={sid for d in items for sid in d['answer_session_ids']};rest=[d for d in D if d['query_id'] not in qids]
  bench.lucid_split=lambda: ([],items,rest)
  old=bench.lucid_other
  bench.lucid_other=lambda rest,n_sess,seed,avoid_arg=():old(rest,n_sess,seed,avoid=avoid)
  return {'fold':index,'folds':len(folds),'query_rows':len(items),'query_ids':[d['query_id'] for d in items]}
 return {}

def main(spec):
 c=METHODS[spec['method']];configure(c);start=time.time();meta={}
 if spec.get('full'):meta=full_data(spec['bed'].split(',')[0],spec.get('fold',0))
 if spec['method'].startswith('legacy_'):result=run_legacy(c)
 elif spec.get('answer'):
  import answer
  answer.bench=bench
  # Original answer.py CLI and its batching/judge logic execute unchanged.
  import runpy
  sys.argv=['answer.py',c['design'],'--mix',c['mix'],'--w',str(c['w']),'--pv',str(c['pv']),'--ctrl',c['ctrl'],'--beds',spec['bed']]
  before=(R/'answer_runs.jsonl').read_text().splitlines() if (R/'answer_runs.jsonl').exists() else []
  runpy.run_path(str(R/'answer.py'),run_name='__main__')
  result=json.loads((R/'answer_runs.jsonl').read_text().splitlines()[-1])
 else:
  result=bench.run(c['design'],'test',set(spec['bed'].split(',')))
 ret.save()
 value={'spec':spec,'config':c,'result':result,'full_data_metadata':meta,'seconds':time.time()-start,'lm_stats':lm.STATS,'snapshot_commit':'d62aa408a4e0b9515eed9313663a025a07b08497','finished':time.strftime('%Y-%m-%d %H:%M:%S')}
 dump(HERE/'results'/(spec['name']+'.json'),value);log('job_complete',name=spec['name'],seconds=value['seconds']);print(json.dumps(value),flush=True)
if __name__=='__main__':main(json.loads(sys.argv[1]))
