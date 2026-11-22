"""Real structured-decoding and prompt-fidelity checks on four development examples."""
from common import *
from transformers import AutoTokenizer
import urllib.error
tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3-8B'),local_files_only=True)
messages=[{'role':'system','content':'You are a careful assistant.'},{'role':'user','content':'Reply OK.'}]
custom=(ROOT/'deployment/qwen3_non_thinking.jinja').read_text()
assert tok.apply_chat_template(messages,tokenize=True,enable_thinking=False)==tok.apply_chat_template(messages,tokenize=True,chat_template=custom)
model_info=json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/models'));assert MODEL in [m['id'] for m in model_info['data']]
T=Transport('preflight-original-prompts');tasks,_,_=load_data();rows=[]
ids={134,155,159,192}
schema={'type':'array','minItems':1,'maxItems':1,'items':{'type':'object','properties':{'id':{'enum':[0]},'note':{'type':'string'}},'required':['id','note']}}
for t in tasks:
 if t['task_id'] not in ids:continue
 prompt='For each request below, imagine earlier conversations with the same user. Write one or two sentences that the user may have said earlier about themselves that would change how a good assistant answers the request. Reply with a JSON list of objects {"id": <id>, "note": <text>}.\n\nID 0: '+t['query']
 result=T.json(prompt,schema=schema);assert len(result)==1 and result[0]['id']==0
 rows.append({'task_id':t['task_id'],'query':t['query'],'generated_note':result[0]['note'],'exact_request_copy':result[0]['note'].strip()==t['query'].strip()})
with sqlite3.connect(T.path) as c:
 for req,resp in c.execute('select request,response from responses'):
  request=json.loads(req);assert len(request['messages'])==2
  assert request['messages'][0]['content']=='You are a careful assistant. Reply with JSON only.'
  assert 'response_format' in request and request['response_format']['type']=='json_schema'
  assert json.loads(resp)['choices'][0]['finish_reason']=='stop'
assert sum(r['exact_request_copy'] for r in rows)==0,'HyDE still copying requests: inspect semantics before benchmarking'
dump(HERE/'prompt-fidelity-preflight.json',{'template_tokens_equal':True,'original_system_message_preserved':True,'schema_only_in_response_format':True,'real_guided_array_decoding':True,'development_examples':rows})
event('server_preflight_pass',examples=4,copy_count=0)
