"""Development ablations then frozen all-125 InMind retrieval diagnostics."""
from common import *
import copy, jsonschema
sys.path[:0]=[str(ROOT/'rsi-memory'),str(ROOT/'external-memory')]
import bridge
OUT=HERE/'retrieval';OUT.mkdir(exist_ok=True)
T=Transport('retrieval');tasks,bg,meta=load_data();dev=set(dev_ids(tasks))
def generate(items,template,field,label):
 result=[]
 def one(ch):
  numbered='\n'.join('ID '+str(i)+': '+x for i,x in enumerate(ch))
  try:
   schema={'type':'array','minItems':len(ch),'maxItems':len(ch),'items':{'type':'object','properties':{'id':{'enum':list(range(len(ch)))+[str(i) for i in range(len(ch))]},field:({'type':'array','items':{'type':'string'},'maxItems':8} if field=='p' else {'type':'string'})},'required':['id',field]}}
   if len(ch)==1:schema=schema['items']
   j=T.json(template.format(lines=numbered,reqs=numbered),schema=schema)
   if isinstance(j,dict):j=[j]
   if not isinstance(j,list):raise ValueError('invalid batch shape')
   d={int(x['id']):x[field] for x in j}
   if set(d)!=set(range(len(ch))) or len(j)!=len(ch):raise ValueError('missing/duplicate IDs')
   vals=[d[i] for i in range(len(ch))]
   if field=='p':
    if not all(isinstance(v,list) and all(isinstance(x,str) for x in v) for v in vals):raise ValueError('invalid property type')
    vals=[v[:8] for v in vals]
   elif not all(isinstance(v,str) for v in vals):raise ValueError('invalid scalar type')
   return vals
  except (ValueError,KeyError,TypeError,jsonschema.ValidationError):
   event('split_batch',kind=label,n=len(ch))
   if len(ch)==1:raise
   m=len(ch)//2;return one(ch[:m])+one(ch[m:])
 for k in range(0,len(items),1):
  result+=one(items[k:k+1])
  if k%40==0:event('generate',kind=label,done=len(result),total=len(items))
 return result
def ranked(s):return np.argsort(-s,kind='stable').tolist()
def rrf(s):
 _,inverse,counts=np.unique(-np.asarray(s),return_inverse=True,return_counts=True)
 starts=np.cumsum(counts)-counts;ranks=starts+(counts-1)/2
 return 1/(60+ranks[inverse])
def prop_vectors(ps):return [embed(p) if p else None for p in ps]
def match(lv,qv):
 if qv is None:return np.zeros(len(lv))
 return np.array([max(0,float((x@qv.T).max())) if x is not None else 0 for x in lv])
def record(task,kind,name,order,lines,target,phase):
 docs=[lines[i] for i in order[:6]];ctx=budget_context(docs)
 return dict(task_id=task['task_id'],domain=task['domain'],kind=kind,method=name,phase=phase,
  hit_at_1=int(target in order[:1]),hit_at_3=int(target in order[:3]),hit_at_6=int(target in order[:6]),
  reciprocal_rank=1/(order.index(target)+1),top6=order[:6],target=target,context=ctx,
  context_target_literal=task['user_message'] in ctx,context_tokens=token_count(ctx))
def summary(rows):
 return {name:{kind:{'n':len(v:=[r for r in rows if r['method']==name and r['kind']==kind]),
   'hit6':sum(r['hit_at_6'] for r in v),'context_literal':sum(r['context_target_literal'] for r in v)}
  for kind in ['query','naive_query']} for name in sorted(set(r['method'] for r in rows))}
# Exact original prompts, no 600-character source truncation; same full chunks for every method.
timelines={t['task_id']:timeline(t) for t in tasks}
source_chunks=[c for ss in timelines.values() for s in ss for c in pairs(s)]
bank=list(dict.fromkeys(text(c) for c in source_chunks))
user_sources={text(c): '\n'.join(m['content'] for m in c if m['role']=='user') for c in source_chunks}
dump(OUT/'bank.json',bank)
active=[i for i,l in enumerate(bank) if user_sources[l].strip()]
values=generate([user_sources[bank[i]] for i in active],bridge.PROPL,'p','memory-properties-user-only')
LP=[[] for _ in bank]
for i,value in zip(active,values):LP[i]=value
dump(OUT/'memory-properties.json',LP)
LV=prop_vectors(LP);LE=embed(bank);lookup={x:i for i,x in enumerate(bank)}
def execute(selected,phase,methods):
 qs=[t[kind] for t in selected for kind in ['query','naive_query']]
 QP=generate(qs,bridge.PROPQ,'p',phase+'-query-properties');QV=prop_vectors(QP)
 gates=generate(qs,bridge.GATEQ,'k',phase+'-gates');assert all(g in ['help','recall'] for g in gates)
 hp='For each request below, imagine earlier conversations with the same user. Write one or two sentences that the user may have said earlier about themselves that would change how a good assistant answers the request. Reply with a JSON list of objects {{"id": <id>, "note": <text>}}.\n\n{reqs}'
 HY=generate(qs,hp,'note',phase+'-hyde');QE=embed(qs,query=True);HE=embed(HY,query=True)
 rows=[]
 for ti,t in enumerate(selected):
  lines=[text(c) for s in timelines[t['task_id']] for c in pairs(s)]
  indices=[lookup[l] for l in lines]; e=LE[indices];lv=[LV[i] for i in indices]
  target=next(i for i,l in enumerate(lines) if t['user_message'] in l and t['assistant_message'] in l)
  for ki,kind in enumerate(['query','naive_query']):
   i=ti*2+ki;dense=e@QE[i];mix=QE[i]+HE[i];mix/=np.linalg.norm(mix);hyde=np.maximum(dense,e@mix)
   both=match(lv,QV[i]);wo=match(lv,QE[i:i+1]);qo=(e@QV[i].T).max(1) if QV[i] is not None else np.zeros(len(e))
   shuffled=[lv[j] for j in np.random.default_rng(42).permutation(len(lv))]
   B={'both':both,'write_only':wo,'query_only':qo,'shuffled':match(shuffled,QV[i])}
   orders={'dense':ranked(dense),'hyde':ranked(hyde)}
   for name,b in B.items():orders[name]=ranked(rrf(dense)+2*rrf(b))
   f=rrf(dense)+2*rrf(both);f[ranked(dense)[0]]+=100
   orders['gated_pin']=ranked(f) if gates[i]=='help' else ranked(dense)
   if 'picker' in methods:
    base=orders['gated_pin'];cand=list(dict.fromkeys(orders['dense'][:6]+base[:6]))
    if gates[i]=='help':
     j=T.json(bridge.PICK.format(q=t[kind],notes='\n'.join(str(n+1)+'. '+lines[x] for n,x in enumerate(cand))),{'type':'object','properties':{'pick':{'type':'array','items':{'type':'integer','minimum':1,'maximum':len(cand)},'maxItems':6}},'required':['pick']})
     order=list(dict.fromkeys([cand[n-1] for n in j['pick']]+base))
    else:order=base
    orders['picker']=order
   for name in methods:rows.append(record(t,kind,name,orders[name],lines,target,phase))
  dump(OUT/(phase+'-results.json'),rows)
  event('retrieval_task',phase=phase,done=ti+1,total=len(selected))
 dump(OUT/(phase+'-summary.json'),summary(rows));return rows
methods=['dense','hyde','write_only','query_only','both','shuffled','gated_pin','picker']
devrows=execute([t for t in tasks if t['task_id'] in dev],'development',methods)
# Select only on development Hit@6, average direct + indirect; ties prefer simpler method.
candidates=['both','gated_pin','picker'];scores={n:sum(r['hit_at_6'] for r in devrows if r['method']==n) for n in candidates}
chosen=max(candidates,key=lambda n:scores[n]);recipe={'generation_isolation':'one note or one query per LLM request; no task or direct/indirect mixing','write_property_input':'user turns only; retrieval preserves full user/assistant chunk','selected':chosen,'scores':scores,'dev_ids':sorted(dev),'w':2,'rrf':60,'k':6,'context_tokens':4096,'rrf_ties':'average rank; equal zero-property scores receive equal bonus','source':sha(__file__),'common_sha256':sha(Path(__file__).with_name('common.py')),'selected_before_test':time.strftime('%Y-%m-%d %H:%M:%S'),'note':'all125 is descriptive because 40 development tasks used for selection; disjoint85 and cross-domain61 primary'}
dump(OUT/'recipe.lock.json',recipe);event('recipe_frozen',**recipe)
test=execute([t for t in tasks if t['task_id'] not in dev],'heldout',['dense','hyde',chosen])
allrows=devrows+test;dump(OUT/'all125-results.json',allrows)
for group,rs in [('all125',allrows),('disjoint85',test),('cross_domain61',[r for r in test if r['domain'] not in {'health_and_wellness','legal','consumer','personal_development','other'}])]:dump(OUT/(group+'-summary.json'),summary(rs))
dump(OUT/'complete.json',{'selected':chosen,'tasks':125,'disjoint':85,'calls':T.calls,'cached':T.cached});event('retrieval_complete',selected=chosen)
