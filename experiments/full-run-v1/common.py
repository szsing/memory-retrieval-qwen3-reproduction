"""Audited local-only transport and benchmark data. No paid provider calls."""
import os, sys, json, hashlib, time, sqlite3, re, urllib.request
from pathlib import Path
import numpy as np
ROOT=Path(os.environ['MEM_PROJECT_ROOT'])
HERE=ROOT/'experiments/full-run-v1'
VENDOR=HERE/'vendor'
sys.path.insert(0,str(VENDOR/'InMind-main/evaluation/scripts'))
from build_timeline import build_timeline, read_jsonl
MODEL='qwen3-8b-b968826-bf16-nothink'
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
 def __init__(self,label):
  self.label=label;self.path=HERE/'llm-cache.sqlite3';self.calls=0;self.cached=0
  with sqlite3.connect(self.path) as c:c.execute('create table if not exists responses (key text primary key, body text)')
 def call(self,messages,max_tokens=2048,schema=None,json_output=False):
  if schema:
   messages=[{'role':'system','content':'Return only a JSON value satisfying this schema: '+json.dumps(schema)}]+messages
  body={'model':MODEL,'messages':messages,'temperature':0,'max_tokens':max_tokens}
  key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
  with sqlite3.connect(self.path) as c:r=c.execute('select body from responses where key=?',(key,)).fetchone()
  if r:self.cached+=1;return json.loads(r[0])
  for attempt in range(3):
   b=dict(body)
   if attempt:b['messages']=messages+[{'role':'user','content':'Return valid complete JSON only. Include every required field.'}]
   req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
   try:
    with urllib.request.urlopen(req,timeout=600) as h:out=json.load(h)
    self.calls+=1
    with (HERE/'transport.jsonl').open('a') as f:f.write(json.dumps({'label':self.label,'key':key,'usage':out['usage'],'finish_reason':out['choices'][0]['finish_reason']})+'\n')
    if out['choices'][0]['finish_reason']!='stop':raise ValueError('truncated model response')
    raw=out['choices'][0]['message']['content'];value=raw
    if schema or json_output:
     m=re.search(r'[\[{].*[\]}]',raw,re.S);value=json.loads(m.group(0) if m else raw)
     if schema:
      import jsonschema;jsonschema.validate(value,schema)
    with sqlite3.connect(self.path) as c:c.execute('insert or replace into responses values (?,?)',(key,json.dumps(value)))
    return value
   except Exception as e:
    if isinstance(e, urllib.error.HTTPError):
     detail=e.read().decode();event('http_error',status=e.code,detail=detail)
     if e.code==400 and 'exceeds fixed context' in detail:raise ValueError(detail)
     raise
    event('transport_retry',label=self.label,reason=str(e),attempt=attempt)
    if attempt==2 and schema and schema.get('type')=='object' and 'id' in schema.get('properties',{}):
     candidate=None
     if isinstance(value,list):
      matching=[item for item in value if isinstance(item,dict) and item.get('id') in schema['properties']['id'].get('enum',[])]
      if matching:candidate=matching[-1]
     if value==[] and 'p' in schema['properties']:candidate={'id':0,'p':[]}
     if candidate is not None:
      import jsonschema
      jsonschema.validate(candidate,schema)
      event('singleton_format_normalized',label=self.label,key=key)
      with (HERE/'format-normalizations.jsonl').open('a') as f:f.write(json.dumps({'key':key,'raw':raw,'normalized':candidate})+'\n')
      with sqlite3.connect(self.path) as c:c.execute('insert or replace into responses values (?,?)',(key,json.dumps(candidate)))
      return candidate
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
def token_count(s):
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
