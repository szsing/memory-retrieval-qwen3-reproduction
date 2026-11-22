import os,json,time,sys,subprocess,traceback
from pathlib import Path
from transport import dump,event
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PY=ROOT/'.venv-baselines/bin/python'
METHODS=['mem0','amem','rootmem','plain','hyde','rerank','full'];deadline=float(os.environ['NATIVE_DEADLINE'])
status={'started':time.strftime('%Y-%m-%d %H:%M:%S'),'deadline':deadline,'phase':'smoke','completed':[],'failed':[],'current':None,'full_benchmark':'inmind','no_other_full_benchmarks':True}
def run(spec):
 if time.time()>=deadline:raise TimeoutError('budget reached')
 name='-'.join(str(spec.get(k,'')) for k in ['stage','bed','method','n']);done=HERE/'done'/(name+'.json')
 if done.exists():return True
 status['current']=spec;dump(HERE/'status.json',status);(HERE/'logs').mkdir(exist_ok=True)
 with (HERE/'logs'/(name+'.log')).open('a') as f:
  code=subprocess.run([str(PY),str(HERE/'worker.py'),json.dumps(spec)],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,NATIVE_JOB=name,RUN_JOB='native-v3-'+name)).returncode
 if code:
  status['failed'].append({'spec':spec,'code':code});dump(HERE/'status.json',status);return False
 dump(done,{'spec':spec,'finished':time.strftime('%Y-%m-%d %H:%M:%S')});status['completed'].append(name);dump(HERE/'status.json',status);return True
try:
 valid=[]
 for m in METHODS[:3]:
  if run({'stage':'smoke','method':m}):valid.append(m)
 status['validated_native_methods']=valid;status['phase']='inmind_subset';dump(HERE/'status.json',status)
 # A failed implementation never becomes a zero score and does not block the other systems.
 active=valid+METHODS[3:]
 for m in active:
  if run({'stage':'retrieve','bed':'inmind','method':m}):run({'stage':'evaluate','bed':'inmind','method':m,'n':61})
 status['phase']='inmind_full';dump(HERE/'status.json',status)
 # Expand after technical validation, regardless of whether a result favors our method.
 for m in active:
  if (HERE/'results'/f'inmind-{m}-61.json').exists():run({'stage':'evaluate','bed':'inmind','method':m,'n':125})
 status['phase']='remaining_subsets';dump(HERE/'status.json',status)
 for bed,n in [('implex',81),('lucid',240)]:
  for m in active:
   if run({'stage':'retrieve','bed':bed,'method':m}):run({'stage':'evaluate','bed':bed,'method':m,'n':n})
 status['phase']='complete'
except TimeoutError:status['phase']='paused_at_budget'
except BaseException:status['phase']='failed';status['error']=traceback.format_exc()
finally:
 status['ended']=time.strftime('%Y-%m-%d %H:%M:%S');dump(HERE/'status.json',status)
 # Existing queue is left paused after the new seven-hour allocation; no unbounded hidden continuation.
