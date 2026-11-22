"""No extra model calls: distinguish property semantics from a lasting-fact prior."""
from common import *
sys.path[:0]=[str(ROOT/'rsi-memory'),str(ROOT/'external-memory')]
import bridge
tasks,_,_=load_data();dev=set(dev_ids(tasks));development_only='--development-only' in sys.argv
if development_only:tasks=[t for t in tasks if t['task_id'] in dev]
prefix='fact-prior-dev' if development_only else 'fact-prior'
bank=json.loads((HERE/'retrieval/bank.json').read_text());lp=json.loads((HERE/'retrieval/memory-properties.json').read_text());lookup={x:i for i,x in enumerate(bank)}
E=embed(bank);rows=[]
def cached(prompt,schema):
 messages=[{'role':'system','content':'Return only a JSON value satisfying this schema: '+json.dumps(schema)},{'role':'system','content':'You are a careful assistant. Reply with JSON only.'},{'role':'user','content':prompt}]
 body={'model':MODEL,'messages':messages,'temperature':0,'max_tokens':2048}
 key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
 with sqlite3.connect(HERE/'llm-cache.sqlite3') as c:r=c.execute('select body from responses where key=?',(key,)).fetchone()
 assert r is not None,'required frozen generator output missing from cache'
 return json.loads(r[0])
def rr(s):
 _,inv,counts=np.unique(-np.asarray(s),return_inverse=True,return_counts=True);starts=np.cumsum(counts)-counts
 return 1/(60+(starts+(counts-1)/2)[inv])
def rank(s):return np.argsort(-s,kind='stable').tolist()
for t in tasks:
 lines=[text(c) for s in timeline(t) for c in pairs(s)];ix=[lookup[l] for l in lines];e=E[ix];has=np.array([bool(lp[i]) for i in ix],float)
 target=next(i for i,l in enumerate(lines) if t['user_message'] in l and t['assistant_message'] in l)
 for kind in ['query','naive_query']:
  q=t[kind];numbered='ID 0: '+q
  ks={'type':'object','properties':{'id':{'enum':[0,'0']},'k':{'type':'string'}},'required':['id','k']}
  ps={'type':'object','properties':{'id':{'enum':[0,'0']},'p':{'type':'array','items':{'type':'string'},'maxItems':8}},'required':['id','p']}
  gate=cached(bridge.GATEQ.format(reqs=numbered),ks)['k'];props=cached(bridge.PROPQ.format(reqs=numbered),ps)['p']
  dense=e@embed([q],query=True)[0];f=rr(dense)+2*rr(has if props else np.zeros(len(lines)))
  orders={'fact_prior':rank(f)};f[rank(dense)[0]]+=100
  orders['gated_fact_prior']=rank(f) if gate=='help' else rank(dense)
  for name,order in orders.items():rows.append({'task_id':t['task_id'],'kind':kind,'method':name,'hit_at_6':int(target in order[:6]),'context':budget_context([lines[i] for i in order[:6]])})
dump(HERE/'retrieval'/(prefix+'-results.json'),rows)
summary={group:{m:{k:[sum(r['hit_at_6'] for r in rr if r['method']==m and r['kind']==k),sum(r['method']==m and r['kind']==k for r in rr)] for k in ['query','naive_query']} for m in ['fact_prior','gated_fact_prior']} for group,rr in ([('development',rows)] if development_only else [('all125',rows),('disjoint85',[r for r in rows if r['task_id'] not in dev])])}
dump(HERE/'retrieval'/(prefix+'-summary.json'),summary);event('fact_prior_control_complete',extra_model_calls=0)

