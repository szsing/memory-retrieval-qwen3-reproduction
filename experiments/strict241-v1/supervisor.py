import os,time,json,subprocess
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1];V3=ROOT/'experiments/native-source-aligned-v3';deadline=float((V3/'deadline.txt').read_text());os.environ.update(NATIVE_DEADLINE=str(deadline),NIGHT_DEADLINE_UNIX=str(deadline))
queue=['full','rootmem','mem0','amem','hyde','rerank','plain'];running={};completed=[];failed=[]
def status():
 tmp=EXP/'status.tmp';tmp.write_text(json.dumps({'updated':time.strftime('%Y-%m-%d %H:%M:%S'),'deadline':deadline,'running':{m:p.pid for m,p in running.items()},'remaining':queue,'completed':completed,'failed':failed},indent=2));tmp.replace(EXP/'status.json')
(EXP/'logs').mkdir(exist_ok=True)
while (queue or running) and time.time()<deadline:
 while queue and len(running)<4:
  m=queue.pop(0)
  if (EXP/'results'/f'{m}-125.json').exists():completed.append(m);continue
  log=(EXP/'logs'/(m+'.log')).open('a');env=dict(os.environ,NATIVE_JOB='strict241-'+m,RUN_JOB='native-v3-strict241-'+m)
  running[m]=subprocess.Popen([str(ROOT/'.venv-baselines/bin/python'),str(EXP/'run_method.py'),m],stdout=log,stderr=subprocess.STDOUT,env=env);log.close()
 for m,p in list(running.items()):
  code=p.poll()
  if code is not None:
   (completed if code==0 else failed).append(m);del running[m]
 status();time.sleep(5)
status()
