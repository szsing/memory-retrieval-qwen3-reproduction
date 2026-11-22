"""Validate format fixes and live upstream-sized batches before any scientific run."""
import os,sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;SRC=HERE/'snapshot';R=SRC/'rsi-memory'
os.environ.update(MEM_LM='openai',MEM_MODEL='qwen3-8b-b968826-bf16-nothink-vllm011',MEM_BASE_URL='http://127.0.0.1:8000/v1',MEM_API_KEY='local-only',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',RUN_JOB='preflight-live')
sys.path[:0]=[str(R),str(SRC/'external-memory')]
from format_adapter import contract,normalize,consumer_value
lock=json.loads((HERE/'upstream-source-lock.json').read_text())
assert all(hashlib.sha256((SRC/p).read_bytes()).hexdigest()==h for p,h in lock.items())
c=contract('Reply with "p"\nID 0: a\nID 1: b')
expected=[{'id':0,'p':[]},{'id':1,'p':['magnets']}]
fixtures=[json.dumps(expected),'\n'.join(json.dumps(x) for x in expected),json.dumps({'items':expected}),json.dumps(expected[::-1]),'```json\n'+json.dumps(expected)+'\n```', '[{"id":"0","p":[]},{"id":"1","p":["magnets"]}]']
for x in fixtures:assert normalize(x,c)[0]==expected
rejected=0
for x in ['[{"id":0,"p":[]}]','[{"id":0,"p":[]},{"id":0,"p":[]}]','[{"id":0,"p":[]},{"id":1,"p":"magnets"}]','[{"id":0,"p":[]},{"id":1", "p":[]}]','prefix '+json.dumps(expected)]:
 try:normalize(x,c)
 except Exception:rejected+=1
assert rejected==5
jc=contract('For each item, a hidden fact about the user is given\nItem 1\nItem 2')
for ids in [[1,2],[0,1]]:
 value=normalize(json.dumps([{'id':i,'v':'yes'} for i in ids]),jc)[0]
 assert [x['id'] for x in consumer_value(value,jc)]==[0,1]
from instrument import install
lm=install()
import bridge,rsi,methods,answer
bridge.PV[0]=2
bed=rsi.bed_inmind();lines=bed['lines'][:30];qs=[x['q'] for x in bed['testA']+bed['testB']]
checks={}
checks['properties_30']=len(bridge.props_lines(lines))==30
checks['queries_8']=len(bridge.props_reqs(qs[:8]))==8
checks['gate_20']=len(bridge.gate_reqs(qs[:20]))==20
checks['hyde_31_global_ids']=len(methods.gen_hyde(qs[:31]))==31
# Exercise the original picker format without changing its prompt or candidate numbering.
pick=bridge.jcall(bridge.PICK.format(q=qs[0],notes='\n'.join(f'{i+1}. {s[:300]}' for i,s in enumerate(lines[:18]))))
checks['picker']=isinstance(pick.get('pick'),list)
items='\n\n'.join(f'Item {i+1}\nHidden fact: I wear a pacemaker.\nRequest: Recommend an MRI scan.\nAnswer: Ask your doctor about pacemaker compatibility.' for i in range(10))
judge=bridge.jcall(answer.JUDGE.format(items=items))
checks['judge_10_index_mapping']=[x['id'] for x in judge]==list(range(10))
assert all(checks.values()),checks
# Assert the transport changed only by adding response_format, with the frozen original limits.
calls=[json.loads(x) for x in (HERE/'api-responses.jsonl').read_text().splitlines() if json.loads(x).get('job')=='preflight-live']
assert calls and all(x['request']['max_tokens']==2048 and x['request']['temperature']==0 for x in calls)
assert all(x['response']['choices'][0]['finish_reason']=='stop' for x in calls)
value={'passed':True,'upstream_files_unchanged':len(lock),'checks':checks,'offline_acceptance_cases':len(fixtures),'offline_rejection_cases':rejected,'live_api_calls':len(calls),'max_tokens':2048,'temperature':0,'prompts_unchanged':True,'prior_llm_outputs_reused':False}
(HERE/'preflight.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
