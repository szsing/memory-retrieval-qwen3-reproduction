"""Observe upstream calls/parsing; keep prompts, batching, parsing and fallback unchanged."""
import os,sys,json,re,time,hashlib,threading,urllib.request,traceback,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
_lock=threading.Lock()
class BudgetExpired(SystemExit):
 def __init__(self):super().__init__(75)
def budget():
 if time.time()>=float(os.environ.get('NIGHT_DEADLINE_UNIX','inf')) or shutil.disk_usage(HERE).free<3*1024**3:raise BudgetExpired()
def log(kind,**kw):
 r={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'kind':kind,'job':os.environ.get('RUN_JOB','unknown'),**kw}
 with _lock,(HERE/'events.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def install():
 import lm,rsi,bridge,methods
 original_call=lm.call;original_jcall=rsi.jcall;original_byid=rsi.by_id;method_byid=methods.by_id;method_batch=methods.batched
 def backend(prompt,system):
  budget();body={'model':lm.MODEL,'temperature':0,'max_tokens':2048,'messages':[{'role':'system','content':system},{'role':'user','content':prompt}]}
  # Exactly upstream _openai body: no schema, no altered system prompt, no added seed.
  req=urllib.request.Request(lm.BASE.rstrip('/')+'/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+os.environ.get('MEM_API_KEY','none')})
  begin=time.time()
  try:
   with urllib.request.urlopen(req,timeout=600) as h:out=json.load(h)
  except Exception as e:log('api_error',error=repr(e));raise
  choice=out['choices'][0];key=lm._key(system,prompt);u=out.get('usage',{})
  with _lock,(HERE/'api-responses.jsonl').open('a') as f:f.write(json.dumps({'key':key,'job':os.environ.get('RUN_JOB'),'request':body,'response':out,'seconds':time.time()-begin})+'\n')
  log('api_response',key=key,finish_reason=choice.get('finish_reason'),thinking_text='<think>' in (choice['message'].get('content') or ''),reasoning_content=bool(choice['message'].get('reasoning_content')),usage=u)
  return choice['message']['content'].strip(),u.get('prompt_tokens',0),u.get('completion_tokens',0)
 lm._openai=backend
 def call(prompt,system='You are a helpful assistant.'):
  budget();out=original_call(prompt,system);expected='json' in system.lower() or 'reply with a json' in prompt.lower() or 'reply with json' in prompt.lower()
  if expected:
   try:
    m=re.search(r'[\[{].*[\]}]',out,re.S);json.loads(m.group(0));ok=True
   except Exception:ok=False
   log('json_text_check',key=lm._key(system,prompt),valid_json=ok,chars=len(out),thinking_text='<think>' in out)
  return out
 lm.call=call
 def jcall(prompt,system='You are a careful assistant. Reply with JSON only.'):
  out=original_jcall(prompt,system);log('jcall_return',key=lm._key(system,prompt),returned_none=out is None);return out
 rsi.jcall=jcall;bridge.jcall=jcall
 def byid(j,n,field='q'):
  out=original_byid(j,n,field);log('by_id',family='rsi',field=field,expected=n,missing=sum(v is None for v in out),empty=sum(v==[] for v in out));return out
 rsi.by_id=byid;bridge.by_id=byid
 def mbyid(j,ch,field):
  out=method_byid(j,ch,field);log('by_id',family='methods',field=field,expected=len(ch),missing=sum(v is None for v in out),empty=sum(v==[] for v in out));return out
 methods.by_id=mbyid
 def batched(*args,**kwargs):
  out=method_batch(*args,**kwargs);log('methods_batched_return',n=len(out),returned_none=sum(v is None for v in out));return out
 methods.batched=batched
 log('instrument_installed',backend=lm.BACKEND,model=lm.MODEL,max_tokens=2048,schema_added=False)
 return lm
