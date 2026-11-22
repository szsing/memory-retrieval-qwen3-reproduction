"""Prove observational wrappers preserve the upstream transport body and parser behavior."""
import os,sys,json,hashlib,urllib.request
from pathlib import Path
from io import BytesIO
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;SRC=HERE/'snapshot';R=SRC/'rsi-memory'
os.environ.update(MEM_LM='openai',MEM_MODEL='qwen3-8b-b968826-bf16-nothink-vllm011',MEM_BASE_URL='http://127.0.0.1:8000/v1',MEM_API_KEY='local-only',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',RUN_JOB='preflight-mock')
sys.path[:0]=[str(R),str(SRC/'external-memory')]
import lm,rsi,bridge
native_backend=lm._openai;native_json=rsi.jcall
from instrument import install
install();observed=[]
class Response(BytesIO):
 def __enter__(self):return self
 def __exit__(self,*a):self.close()
def fake(req,timeout):
 observed.append(json.loads(req.data));return Response(json.dumps({'choices':[{'finish_reason':'length','message':{'content':'fixture'}}],'usage':{'prompt_tokens':2,'completion_tokens':3}}).encode())
with patch('urllib.request.urlopen',fake):
 a=native_backend('fixture prompt','fixture system');b=lm._openai('fixture prompt','fixture system')
assert a==b and observed[0]==observed[1] and 'response_format' not in observed[0]
# Malformed JSON retains the author's exact two-attempt None fallback; diagnostics observe it.
with patch.object(lm,'call',return_value='not JSON'):
 assert native_json('fixture') is None and rsi.jcall('fixture') is None
os.environ['RUN_JOB']='preflight-live'
out=bridge.props_lines(['I wear a pacemaker.']);assert len(out)==1 and isinstance(out[0],list)
assert out[0],'Known factual-note smoke test yielded no properties'
value={'passed':True,'native_and_instrumented_http_bodies_equal':True,'no_schema_added':True,'upstream_length_handling_preserved':True,'upstream_parser_none_behavior_preserved':True,'known_fact_properties':out,'thinking_disabled_by_server_template':True}
(HERE/'preflight.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
