"""Overnight budget and strictly immutable prompts, including error paths."""
from common import *
import shutil,copy
from contextlib import contextmanager
@contextmanager
def cache_connection(path):
 c=sqlite3.connect(path,timeout=60)
 try:
  with c:yield c
 finally:c.close()
class NightBudgetExpired(SystemExit):
 def __init__(self,reason):self.reason=reason;super().__init__(75)
def check_budget():
 deadline=float(os.environ.get('NIGHT_DEADLINE_UNIX','inf'))
 if time.time()>=deadline:raise NightBudgetExpired('overnight wall-clock budget reached')
 if shutil.disk_usage(HERE).free<3*1024**3:raise NightBudgetExpired('preserve 3 GiB free disk for results and checkpoints')
class NightTransport(Transport):
 def _generate_or_load(self,body,key,schema,json_output,response_format):
  import jsonschema
  with cache_connection(self.path) as c:hit=c.execute('select value,request from responses where key=?',(key,)).fetchone()
  # Reject an earlier retry if it changed messages. Never attach JSON instructions to prose readers.
  if hit and json.loads(hit[1])==body:
   with self._stats_lock:self.cached+=1
   return json.loads(hit[0])
  check_budget();start=time.perf_counter()
  req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=600) as h:out=json.load(h)
  with self._stats_lock:self.calls+=1
  with self._trace_lock,(HERE/('transport-'+str(os.getpid())+'.jsonl')).open('a') as f:f.write(json.dumps({'key':key,'label':self.label,'attempt':0,'request':body,'response':out,'seconds':time.perf_counter()-start,'strict_prompt_identity':True})+'\n')
  choice=out['choices'][0]
  if choice['finish_reason']!='stop':raise ValueError('Output did not finish; no prompt mutation or invalid output accepted: '+choice['finish_reason'])
  raw=choice['message']['content'];value=raw
  if schema is not None or json_output or response_format is not None:
   m=re.search(r'[\[{].*[\]}]',raw,re.S);value=json.loads(m.group(0) if m else raw)
   if schema is not None:jsonschema.validate(value,schema)
  with cache_connection(self.path) as c:c.execute('insert or replace into responses values (?,?,?,?)',(key,json.dumps(value),json.dumps(body),json.dumps(out)))
  return value
