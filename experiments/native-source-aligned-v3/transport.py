"""Same local model, immutable native prompts, validated format-only decoding."""
import os,json,time,hashlib,sqlite3,threading,urllib.request
from pathlib import Path
from contextlib import contextmanager
HERE=Path(__file__).resolve().parent
MODEL='qwen3-8b-b968826-bf16-nothink-vllm011'
class InvalidNative(BaseException):pass
class BudgetEnd(BaseException):pass
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def budget():
 if time.time()>float(os.environ.get('NATIVE_DEADLINE','inf')):raise BudgetEnd()
def event(kind,**kw):
 with (HERE/'events.jsonl').open('a') as f:f.write(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'job':os.environ.get('NATIVE_JOB'),'kind':kind,**kw},ensure_ascii=False)+'\n')
@contextmanager
def db():
 c=sqlite3.connect(HERE/'llm.sqlite3',timeout=120)
 try:
  with c:yield c
 finally:c.close()
class Transport:
 def __init__(self):
  self.calls=0;self.cached=0;self.lock=threading.Lock()
  with db() as c:c.execute('create table if not exists outputs(key text primary key,value text)')
 def call(self,messages,max_tokens=2048,schema=None,json_output=False,response_format=None):
  import jsonschema
  budget();body={'model':MODEL,'temperature':0,'max_tokens':max_tokens,'messages':messages}
  if schema:body['response_format']={'type':'json_schema','json_schema':{'name':'output','schema':schema,'strict':True}}
  elif response_format:body['response_format']=response_format
  elif json_output:body['response_format']={'type':'json_object'}
  key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
  with self.lock:
   with db() as c:hit=c.execute('select value from outputs where key=?',(key,)).fetchone()
   if hit:self.cached+=1;return json.loads(hit[0])
   for attempt in range(2):
    budget();start=time.time()
    try:
     req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
     with urllib.request.urlopen(req,timeout=600) as h:res=json.load(h)
     self.calls+=1;choice=res['choices'][0];raw=choice['message']['content'] or ''
     with (HERE/'api-responses.jsonl').open('a') as f:f.write(json.dumps({'job':os.environ.get('NATIVE_JOB'),'key':key,'attempt':attempt,'request':body,'response':res,'seconds':time.time()-start})+'\n')
     event('api_response',finish=choice['finish_reason'],thinking=bool('<think>' in raw or choice['message'].get('reasoning_content')))
     if choice['finish_reason']!='stop':raise ValueError('Truncated/non-stop response')
     if '<think>' in raw or choice['message'].get('reasoning_content'):raise ValueError('Unexpected thinking')
     value=json.loads(raw) if body.get('response_format') else raw.strip()
     if schema:jsonschema.validate(value,schema)
     with db() as c:c.execute('insert or replace into outputs values (?,?)',(key,json.dumps(value)))
     return value
    except Exception as e:
     event('attempt_failed',attempt=attempt,error=repr(e))
     if attempt==1:raise InvalidNative(str(e))
  raise InvalidNative('unreachable')
