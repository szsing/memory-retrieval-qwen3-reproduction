"""Audited local-only transport and benchmark data. No paid provider calls."""
import os, sys, json, hashlib, time, sqlite3, re, urllib.request, threading, fcntl
from pathlib import Path
import numpy as np
ROOT=Path(os.environ['MEM_PROJECT_ROOT'])
HERE=Path(__file__).resolve().parent
VENDOR=HERE/'vendor'
sys.path.insert(0,str(VENDOR/'InMind-main/evaluation/scripts'))
from build_timeline import build_timeline, read_jsonl
MODEL='qwen3-8b-b968826-bf16-nothink-vllm011'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MEM0_TELEMETRY='false',OPENAI_API_KEY='local-only',OPENAI_BASE_URL='http://127.0.0.1:8000/v1',TOKENIZERS_PARALLELISM='false')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
def event(stage,**kw):
 x={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':stage,**kw};print(json.dumps(x),flush=True)
def load_data():
 d=VENDOR/'InMind-main';tasks=read_jsonl(d/'benchmark/dataset/inmind.jsonl');bg=read_jsonl(d/'evaluation/background/lme_s_background.jsonl');meta=json.loads((d/'evaluation/background/manifest.json').read_text());return tasks,bg,meta
def timeline(task):
 _,bg,meta=load_data();return build_timeline(task,bg,meta)['sessions']
def pairs(session):
 """Contiguous interaction chunks; no crossing session boundaries, no gold fields."""
 chunks=[];current=[]
 for t in session['turns']:
  if t['role']=='user' and current:chunks.append(current);current=[]
  current.append({'role':t['role'],'content':t['content']})
 if current:chunks.append(current)
 return chunks
def text(messages):return '\n'.join(m['role'].capitalize()+': '+m['content'] for m in messages)
def dev_ids(tasks):
 domains={'health_and_wellness','legal','consumer','personal_development','other'}
 return [t['task_id'] for t in sorted([t for t in tasks if t['domain'] in domains],key=lambda t:int(hashlib.md5(str(t['task_id']).encode()).hexdigest()[:8],16))[:40]]
class Transport:
 """Original messages are immutable; JSON constraints are API decoding parameters."""
 def __init__(self,label):
  self.label=label;self.path=HERE/'llm-cache.sqlite3';self.calls=0;self.cached=0;self._stats_lock=threading.Lock();self._trace_lock=threading.Lock();(HERE/'cache-locks').mkdir(exist_ok=True)
  with sqlite3.connect(self.path,timeout=60) as c:c.execute('create table if not exists responses (key text primary key, value text, request text, response text)')
 def call(self,messages,max_tokens=2048,schema=None,json_output=False,response_format=None):
  import copy, jsonschema
  body={'model':MODEL,'messages':copy.deepcopy(messages),'temperature':0,'max_tokens':max_tokens,'seed':42,'chat_template_kwargs':{'enable_thinking':False}}
  if response_format is not None:body['response_format']=response_format
  elif schema is not None:body['response_format']={'type':'json_schema','json_schema':{'name':'response','schema':schema,'strict':True}}
  # Validation schema is cache metadata; it is never inserted into messages.
  key=hashlib.sha256(json.dumps({'body':body,'validation_schema':schema},sort_keys=True).encode()).hexdigest()
  with (HERE/'cache-locks'/key).open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   return self._generate_or_load(body,key,schema,json_output,response_format)
 def _generate_or_load(self,body,key,schema,json_output,response_format):
  import copy, jsonschema
  with sqlite3.connect(self.path,timeout=60) as c:hit=c.execute('select value from responses where key=?',(key,)).fetchone()
  if hit:
   with self._stats_lock:self.cached+=1
   return json.loads(hit[0])
  for attempt in range(3):
   request_body=copy.deepcopy(body)
   if attempt:request_body['messages'][-1]['content']+='\n\nReturn valid JSON only.'
   req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(request_body).encode(),headers={'Content-Type':'application/json'})
   start=time.perf_counter()
   try:
    with urllib.request.urlopen(req,timeout=600) as h:out=json.load(h)
    with self._stats_lock:self.calls+=1
    with self._trace_lock, (HERE/('transport-'+str(os.getpid())+'.jsonl')).open('a') as f:f.write(json.dumps({'key':key,'label':self.label,'attempt':attempt,'request':request_body,'response':out,'seconds':time.perf_counter()-start})+'\n')
    choice=out['choices'][0]
    if choice['finish_reason']!='stop':raise ValueError('model output truncated: '+choice['finish_reason'])
    raw=choice['message']['content'];value=raw
    if schema is not None or json_output or response_format is not None:
     m=re.search(r'[\[{].*[\]}]',raw,re.S);value=json.loads(m.group(0) if m else raw)
     if schema is not None:jsonschema.validate(value,schema)
    with sqlite3.connect(self.path,timeout=60) as c:c.execute('insert or replace into responses values (?,?,?,?)',(key,json.dumps(value),json.dumps(request_body),json.dumps(out)))
    return value
   except urllib.error.HTTPError as e:
    detail=e.read().decode();event('http_error',status=e.code,detail=detail);raise
   except (ValueError,KeyError,TypeError,jsonschema.ValidationError) as e:
    event('format_failure',label=self.label,key=key,attempt=attempt,error=str(e))
    if attempt==2:raise
  raise RuntimeError('transport exhausted')
 def json(self,prompt,schema=None):return self.call([{'role':'system','content':'You are a careful assistant. Reply with JSON only.'},{'role':'user','content':prompt}],schema=schema,json_output=True)
_emb=None
def embed(texts,query=False):
 global _emb
 if _emb is None:
  from sentence_transformers import SentenceTransformer
  _emb=SentenceTransformer('BAAI/bge-small-en-v1.5',device='cpu',local_files_only=True)
 if query and os.environ.get('BGE_QUERY_PREFIX','0')=='1':texts=['Represent this sentence for searching relevant passages: '+t for t in texts]
 return _emb.encode(texts,normalize_embeddings=True,show_progress_bar=False)
_tok_lock=threading.Lock()
def token_count(s):
 with _tok_lock:return _token_count(s)
def _token_count(s):
 from transformers import AutoTokenizer
 global _tok
 if '_tok' not in globals():_tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3-8B'),local_files_only=True)
 return len(_tok.encode(s,add_special_tokens=False))
def budget_context(documents,max_tokens=4096):
 kept=[]
 for d in documents:
  candidate='\n\n'.join(kept+[d])
  if token_count(candidate)>max_tokens:break
  kept.append(d)
 return '\n\n'.join(kept)
