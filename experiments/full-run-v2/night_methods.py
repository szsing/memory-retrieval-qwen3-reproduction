"""Frozen raw-note retrieval methods and cacheable single-input feature generation."""
from night_data import *
sys.path[:0]=[str(ROOT/'rsi-memory'),str(ROOT/'external-memory')]
import bridge
from night_budget import NightTransport
T=NightTransport('night-features')
HYDE='For each request below, imagine earlier conversations with the same user. Write one or two sentences that the user may have said earlier about themselves that would change how a good assistant answers the request. Reply with a JSON list of objects {{"id": <id>, "note": <text>}}.\n\n{reqs}'
def cached(key,fn):
 with connect() as c:r=c.execute('select value from features where key=?',(key,)).fetchone()
 if r:return json.loads(r[0])
 value=fn()
 with connect() as c:c.execute('insert or ignore into features values (?,?)',(key,json.dumps(value)))
 return value
def feature(s,template,field):
 schema={'type':'array','minItems':1,'maxItems':1,'items':{'type':'object','properties':{'id':{'enum':[0]},field:({'type':'array','items':{'type':'string'},'maxItems':8} if field=='p' else {'type':'string'})},'required':['id',field]}}
 return T.json(template.format(lines='ID 0: '+s,reqs='ID 0: '+s),schema)[0][field]
def vec(strings):
 # Binary embedding store avoids large JSON expansions; same exact full-text embeddings.
 keys=[hashlib.sha256(('bge-small-full-v1:'+s).encode()).hexdigest() for s in strings]
 with connect() as c:
  c.execute('create table if not exists embeddings(key text primary key,value blob)')
  found={k:v for k,v in c.execute('select key,value from embeddings where key in ('+','.join('?' for _ in keys)+')',keys)} if keys else {}
 missing=list(dict.fromkeys((k,s) for k,s in zip(keys,strings) if k not in found))
 if missing:
  vals=embed([s for k,s in missing]).astype('float32')
  with connect() as c:c.executemany('insert or ignore into embeddings values (?,?)',[(k,v.tobytes()) for (k,s),v in zip(missing,vals)])
  found.update({k:v.tobytes() for (k,s),v in zip(missing,vals)})
 return np.array([np.frombuffer(found[k],dtype='float32') for k in keys]) if keys else None
def props(s,write=False):
 if not s.strip():return []
 return feature(s,bridge.PROPL if write else bridge.PROPQ,'p')
def ranked(s):return np.argsort(-s,kind='stable').tolist()
def rrf(s):
 _,inv,c=np.unique(-np.asarray(s),return_inverse=True,return_counts=True);starts=np.cumsum(c)-c
 return 1/(60+(starts+(c-1)/2)[inv])
def match(lv,qv):
 if qv is None:return np.zeros(len(lv))
 return np.array([max(0,float((x@qv.T).max())) if x is not None else 0 for x in lv])
def orders(notes,queries,method):
 E=vec([n['text'] for n in notes]);QE=vec([q['text'] for q in queries]);result=[]
 lv=None
 if method=='properties':
  ps=[props(n['user_text'],True) for n in notes]
  lv=[vec(p) for p in ps];shuffled=[lv[j] for j in np.random.default_rng(42).permutation(len(lv))]
 for i,q in enumerate(queries):
  d=E@QE[i];base=ranked(d)
  if method=='dense':out={'dense':base}
  elif method=='hyde':
   h=vec([feature(q['text'],HYDE,'note')])[0];mix=QE[i]+h;mix/=np.linalg.norm(mix);out={'hyde':ranked(np.maximum(d,E@mix))}
  elif method=='dense_pick':
   cand=base[:12];schema={'type':'object','properties':{'pick':{'type':'array','items':{'type':'integer','minimum':1,'maximum':len(cand)},'maxItems':6}},'required':['pick']}
   j=T.json(bridge.PICK.format(q=q['text'],notes='\n'.join(str(j+1)+'. '+notes[n]['text'] for j,n in enumerate(cand))),schema)
   out={'dense_pick':list(dict.fromkeys([cand[n-1] for n in j['pick']]+base))}
  else:
   qv=vec(props(q['text']));g=feature(q['text'],bridge.GATEQ,'k');assert g in ('help','recall')
   b=match(lv,qv);wo=match(lv,QE[i:i+1]);qo=(E@qv.T).max(1) if qv is not None else np.zeros(len(E))
   pri=np.array([int(x is not None) for x in lv]);out={}
   for name,v in [('both',b),('write_only',wo),('query_only',qo),('shuffled',match(shuffled,qv)),('fact_prior',pri)]:out[name]=ranked(rrf(d)+2*rrf(v))
   pin=rrf(d)+2*rrf(b);pin[base[0]]+=100
   out['gated_pin']=ranked(pin) if g=='help' else base
   out['gated_fact_prior']=out['fact_prior'] if g=='help' else base
   # Picker ablation follows original candidate pool and its own fallback.
   if g=='help':
    cand=list(dict.fromkeys(base[:6]+out['gated_pin'][:6]));schema={'type':'object','properties':{'pick':{'type':'array','items':{'type':'integer','minimum':1,'maximum':len(cand)},'maxItems':6}},'required':['pick']}
    j=T.json(bridge.PICK.format(q=q['text'],notes='\n'.join(str(j+1)+'. '+notes[n]['text'] for j,n in enumerate(cand))),schema)
    out['picker']=list(dict.fromkeys([cand[n-1] for n in j['pick']]+out['gated_pin']))
   else:out['picker']=out['gated_pin']
  result.append(out)
 return result
