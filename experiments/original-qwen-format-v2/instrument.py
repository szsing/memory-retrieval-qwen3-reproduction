"""Strict output-format adapter; source prompts, batch sizes, budgets and ranking are unchanged."""
import os,json,time,hashlib,threading,urllib.request,sqlite3,fcntl,shutil
from pathlib import Path
from contextlib import contextmanager
from format_adapter import contract,normalize,consumer_value,InvalidOutput
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'original-qwen-night-v1';_lock=threading.Lock();_legacy=None
class BudgetExpired(SystemExit):
 def __init__(self):super().__init__(75)
class FormatFailure(SystemExit):
 def __init__(self,reason):self.reason=reason;super().__init__(78)
def budget():
 if time.time()>=float(os.environ.get('NIGHT_DEADLINE_UNIX','inf')) or shutil.disk_usage(HERE).free<3*1024**3:raise BudgetExpired()
def log(kind,**kw):
 with _lock,(HERE/'events.jsonl').open('a') as f:f.write(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'kind':kind,'job':os.environ.get('RUN_JOB','unknown'),**kw},ensure_ascii=False)+'\n')
@contextmanager
def db():
 c=sqlite3.connect(HERE/'format-cache.sqlite3',timeout=120)
 try:
  with c:yield c
 finally:c.close()
def legacy():
 global _legacy
 with _lock:
  if _legacy is None:
   _legacy={};path=OLD/'api-responses.jsonl'
   if path.exists():
    for line in path.open():
     r=json.loads(line)
     if str(r.get('job','')).startswith('preflight'):continue
     _legacy.setdefault(r['key'],[]).append(r)
 return _legacy

def install():
 import lm,rsi,bridge,methods
 with db() as c:c.execute('create table if not exists outputs(key text primary key,value text,request text,response text)')
 (HERE/'locks').mkdir(exist_ok=True)
 old_byid=rsi.by_id;old_mbyid=methods.by_id
 def call(prompt,system='You are a helpful assistant.'):
  budget();c=contract(prompt);base={'model':lm.MODEL,'temperature':0,'max_tokens':2048,'messages':[{'role':'system','content':system},{'role':'user','content':prompt}]};body=dict(base)
  if c:body['response_format']={'type':'json_schema','json_schema':{'name':'output','schema':c['schema'],'strict':True}}
  key=hashlib.sha256(json.dumps({'adapter':2,'body':body},sort_keys=True).encode()).hexdigest()
  with (HERE/'locks'/key).open('w') as guard:
   fcntl.flock(guard,fcntl.LOCK_EX)
   with db() as cx:hit=cx.execute('select value from outputs where key=?',(key,)).fetchone()
   if hit:
    lm.STATS['cached']+=1;out=hit[0];log('cache_hit',key=key)
    if c:log('json_text_check',key=key,valid_json=True)
    return out
   # Recover only exact same requests, complete responses and validated ID/field mappings.
   for r in (legacy().get(lm._key(system,prompt),[]) if os.environ.get('REUSE_VALIDATED_V1')=='1' else []):
    if r['request']!=base or r['response']['choices'][0]['finish_reason']!='stop':continue
    raw=r['response']['choices'][0]['message']['content']
    try:
     if c:obj,changes=normalize(raw,c);out=json.dumps(obj,ensure_ascii=False)
     else:out=raw.strip();changes=[]
    except Exception as e:log('legacy_rejected',key=key,reason=str(e)[:300]);continue
    with db() as cx:cx.execute('insert or replace into outputs values (?,?,?,?)',(key,out,json.dumps(body),json.dumps(r['response'])))
    lm.STATS['cached']+=1;log('legacy_recovered',key=key,normalizations=changes,kind_name=c['kind'] if c else 'prose')
    if c:log('json_text_check',key=key,valid_json=True)
    return out
   for attempt in range(2):
    budget();req=urllib.request.Request(lm.BASE.rstrip('/')+'/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+os.environ.get('MEM_API_KEY','none')})
    try:
     start=time.time()
     with urllib.request.urlopen(req,timeout=600) as h:response=json.load(h)
     choice=response['choices'][0];usage=response.get('usage',{});lm.STATS['calls']+=1;lm.STATS['tok_in']+=usage.get('prompt_tokens',0);lm.STATS['tok_out']+=usage.get('completion_tokens',0)
     with _lock,(HERE/'api-responses.jsonl').open('a') as f:f.write(json.dumps({'key':key,'job':os.environ.get('RUN_JOB'),'attempt':attempt,'request':body,'response':response,'seconds':time.time()-start})+'\n')
     raw=choice['message']['content'];thinking='<think>' in (raw or '') or bool(choice['message'].get('reasoning_content'))
     log('api_response',key=key,finish_reason=choice.get('finish_reason'),thinking_text=thinking,reasoning_content=bool(choice['message'].get('reasoning_content')))
     if choice['finish_reason']!='stop':raise InvalidOutput('Truncated/non-stop output; no token-limit changes allowed')
     if thinking:raise InvalidOutput('Unexpected thinking output')
     if c:obj,changes=normalize(raw,c);out=json.dumps(obj,ensure_ascii=False);log('json_text_check',key=key,valid_json=True)
     else:out=raw.strip()
     with db() as cx:cx.execute('insert or replace into outputs values (?,?,?,?)',(key,out,json.dumps(body),json.dumps(response)))
     return out
    except Exception as e:
     log('format_attempt_failed',key=key,attempt=attempt,error=str(e)[:700])
     if attempt==1:log('format_failure',key=key);raise FormatFailure(str(e))
  raise FormatFailure('unreachable')
 lm.call=call
 def jcall(prompt,system='You are a careful assistant. Reply with JSON only.'):
  c=contract(prompt)
  if c is None:raise FormatFailure('Unrecognized JSON-call contract')
  obj,changes=normalize(lm.call(prompt,system),c);out=consumer_value(obj,c)
  if c['kind']=='judge':log('judge_index_normalization',input_ids=c['ids'],consumer_ids=[x['id'] for x in out])
  log('jcall_return',returned_none=False);return out
 rsi.jcall=jcall;bridge.jcall=jcall
 def byid(j,n,field='q'):
  out=old_byid(j,n,field)
  if len(out)!=n or any(v is None for v in out):log('format_failure',reason='Missing downstream rsi item');raise FormatFailure('Missing downstream item')
  log('by_id',family='rsi',field=field,expected=n,missing=0,empty=sum(v==[] for v in out));return out
 rsi.by_id=byid;bridge.by_id=byid
 def mbyid(j,ch,field):
  out=old_mbyid(j,ch,field)
  if len(out)!=len(ch) or any(v is None for v in out):log('format_failure',reason='Missing downstream methods item');raise FormatFailure('Missing downstream item')
  log('by_id',family='methods',field=field,expected=len(ch),missing=0,empty=sum(v==[] for v in out));return out
 methods.by_id=mbyid
 log('instrument_installed',adapter_version=2,max_tokens=2048,schema_added_at_api_only=True,prompts_changed=False,batch_sizes_changed=False)
 return lm
