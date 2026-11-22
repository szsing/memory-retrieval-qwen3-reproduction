"""Independent dense top-12 LLM rerank control; no property candidates or fallback."""
from common import *
sys.path[:0]=[str(ROOT/'rsi-memory'),str(ROOT/'external-memory')]
import bridge
T=Transport('dense-pick-control');tasks,_,_=load_data();dev=set(dev_ids(tasks));rows=[]
out=HERE/'retrieval/dense-pick-control.json'
if out.exists():rows=json.loads(out.read_text())
done={(r['task_id'],r['kind']) for r in rows}
for t in tasks:
 lines=[text(c) for s in timeline(t) for c in pairs(s)];E=embed(lines)
 for kind in ['query','naive_query']:
  if (t['task_id'],kind) in done:continue
  q=embed([t[kind]],query=True)[0];order=np.argsort(-(E@q),kind='stable').tolist();cand=order[:12]
  j=T.json(bridge.PICK.format(q=t[kind],notes='\n'.join(str(i+1)+'. '+lines[n] for i,n in enumerate(cand))),{'type':'object','properties':{'pick':{'type':'array','items':{'type':'integer','minimum':1,'maximum':len(cand)},'maxItems':6}},'required':['pick']})
  ranked=list(dict.fromkeys([cand[n-1] for n in j['pick']]+order));ctx=budget_context([lines[n] for n in ranked[:6]])
  target=next(i for i,s in enumerate(lines) if t['user_message'] in s and t['assistant_message'] in s)
  rows.append({'task_id':t['task_id'],'kind':kind,'method':'dense_pick','context':ctx,'context_tokens':token_count(ctx),'hit_at_6':int(target in ranked[:6]),'top6':ranked[:6]});dump(out,rows)
 event('dense_pick_task',task=t['task_id'],rows=len(rows),total=250)
summary={name:{k:[sum(r['hit_at_6'] for r in rr if r['kind']==k),sum(r['kind']==k for r in rr)] for k in ['query','naive_query']} for name,rr in [('all125',rows),('disjoint85',[r for r in rows if r['task_id'] not in dev])]}
dump(HERE/'retrieval/dense-pick-summary.json',summary)

import run_fact_prior_control

