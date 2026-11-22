"""Focused regression: prose truncation cannot turn into a JSON prompt retry."""
from night_budget import *
from unittest.mock import patch
from io import BytesIO

def main():
 t=NightTransport('transport-regression');msgs=[{'role':'user','content':'Give pet advice.'}];original=copy.deepcopy(msgs)
 body={'model':MODEL,'messages':msgs,'max_tokens':2048};key='preflight-only-'+str(time.time_ns())
 responses=[]
 class Response(BytesIO):
  def __enter__(self):return self
  def __exit__(self,*a):self.close()
 def fake(req,timeout):
  responses.append(json.loads(req.data))
  return Response(json.dumps({'choices':[{'finish_reason':'length','message':{'content':'unfinished answer'}}]}).encode())
 with patch('urllib.request.urlopen',fake):
  try:t._generate_or_load(body,key,None,False,None)
  except ValueError:pass
  else:raise AssertionError('truncated prose accepted')
 assert len(responses)==1 and msgs==original and responses[0]['messages']==original
 with sqlite3.connect(t.path) as c:assert not c.execute('select 1 from responses where key=?',(key,)).fetchone()
 old=os.environ.get('NIGHT_DEADLINE_UNIX');os.environ['NIGHT_DEADLINE_UNIX']='0'
 try:
  try:check_budget()
  except NightBudgetExpired as e:assert e.code==75
  else:raise AssertionError('deadline ignored')
 finally:
  if old is None:os.environ.pop('NIGHT_DEADLINE_UNIX')
  else:os.environ['NIGHT_DEADLINE_UNIX']=old
 dump(HERE/'night-transport-check.json',{'passed':True,'truncated_prose_not_cached':True,'prompts_never_mutated':True,'no_json_retry_for_prose':True,'deadline_exit75':True})
if __name__=='__main__':main()
