"""Offline format fixtures; isolated temporary cache; never calls the model service."""
import tempfile, json
from pathlib import Path
import common
from unittest.mock import patch
schema={'type':'object','properties':{'id':{'enum':[0]},'k':{'enum':['help','recall']}},'required':['id','k']}
class Reply:
 def __init__(self,value):self.value=value
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def read(self):return json.dumps({'choices':[{'message':{'content':json.dumps(self.value)},'finish_reason':'stop'}],'usage':{'prompt_tokens':1,'completion_tokens':1}}).encode()
with tempfile.TemporaryDirectory() as directory:
 common.HERE=Path(directory);t=common.Transport('offline-fixture')
 with patch('urllib.request.urlopen',return_value=Reply([{'id':0,'k':'help'}])) as mock:
  x=t.json('singleton-array',schema);assert x=={'id':0,'k':'help'} and mock.call_count==3
  assert t.json('singleton-array',schema)==x and mock.call_count==3
 with patch('urllib.request.urlopen',return_value=Reply([{'id':1,'k':'help'}])):
  try:t.json('wrong-id',schema)
  except Exception:pass
  else:raise AssertionError('wrong ID silently accepted')
 with patch('urllib.request.urlopen',return_value=Reply([])):
  try:t.json('missing-gate',schema)
  except Exception:pass
  else:raise AssertionError('missing gate silently invented')
 p_schema={'type':'object','properties':{'id':{'enum':[0]},'p':{'type':'array','items':{'type':'string'}}},'required':['id','p']}
 with patch('urllib.request.urlopen',return_value=Reply([])):
  assert t.json('explicit-empty-property',p_schema)=={'id':0,'p':[]}
print('PASS: singleton format compatibility, cache reuse, invalid-ID rejection, missing-gate rejection, explicit empty-property preservation; zero real HTTP calls')

