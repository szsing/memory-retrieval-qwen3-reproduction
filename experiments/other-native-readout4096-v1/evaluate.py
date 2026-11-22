"""Read-only retrieval references; whole native units up to4096 tokens. Old source judge is diagnostic only."""
import os,sys,json,time,traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];V3=ROOT/'experiments/native-source-aligned-v3'
sys.path.insert(0,str(V3));import worker as W
from transport import dump,budget,event
spec=json.loads(sys.argv[1]);bed=spec['bed'];method=spec['method'];n=spec['n'];D=W.prepare()[bed]
os.environ['NATIVE_JOB']='other4096-'+bed+'-'+method
contexts=json.loads((V3/'contexts'/bed/(method+'.json')).read_text());path=OUT/'answers'/bed/(method+'.json');rows=json.loads(path.read_text()) if path.exists() else {}
try:
 for i,t in enumerate(D['tasks'][:n]):
  budget()
  if str(i) in rows:continue
  c=contexts[str(i)];kept=[]
  for unit in c['units']:
   text=str(unit)
   if W.B.token_count('\n'.join('- '+x for x in kept+[text]))>4096:break
   kept.append(text)
  notes='\n'.join('- '+x for x in kept) or '(none)';response=W.lm.call(W.answer.ANS.format(notes=notes,q=t['q']))
  rows[str(i)]={'id':t['id'],'notes':notes,'answer':response,'applied':None,'raw_hit6':c['raw_hit6'],'context_tokens':W.B.token_count(notes),'max_context_tokens':4096,'whole_units_kept':len(kept),'native_units_available':len(c['units'])};dump(path,rows)
  dump(OUT/'progress'/(bed+'-'+method+'.json'),{'answered':len(rows),'total':n,'updated':time.strftime('%F %T')})
 # Retain the source judge only for continuity. These are not official benchmark accuracy scores.
 pending=[i for i in range(n) if rows[str(i)]['applied'] is None]
 for start in range(0,len(pending),10):
  budget();indices=pending[start:start+10]
  items='\n\n'.join(f"Item {j+1}\nHidden fact: {D['tasks'][i]['fact'][:400]}\nRequest: {D['tasks'][i]['q']}\nAnswer: {rows[str(i)]['answer']}" for j,i in enumerate(indices))
  judged=W.bridge.jcall(W.answer.JUDGE.format(items=items));assert [x['id'] for x in judged]==list(range(len(indices)))
  for i,r in zip(indices,judged):rows[str(i)]['applied']=r['v']=='yes'
  dump(path,rows)
 dump(OUT/'results'/f'{bed}-{method}-{n}.json',{'bed':bed,'method':method,'n':n,'complete':True,'diagnostic_source_judge_correct':sum(x['applied'] for x in rows.values()),'metric_warning':'Original simplified judge only; not official benchmark accuracy. Implex/LUCid source subsets are not full benchmarks.','raw_hit6':sum(bool(x['raw_hit6']) for x in rows.values()) if method in ['full','hyde','rerank','plain'] else None})
except BaseException as e:
 dump(OUT/'errors'/(bed+'-'+method+'.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise

