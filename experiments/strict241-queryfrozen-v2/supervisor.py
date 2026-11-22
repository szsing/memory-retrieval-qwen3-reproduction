import os,time,json,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];V3=ROOT/'experiments/native-source-aligned-v3';V1=ROOT/'experiments/strict241-v1';deadline=float((V3/'deadline.txt').read_text())
os.environ.update(NATIVE_DEADLINE=str(deadline),NIGHT_DEADLINE_UNIX=str(deadline))
for sub in ['answers','results']:(OUT/sub).mkdir(exist_ok=True)
references={}
for m in ['mem0','amem','rootmem']:
 for folder,name in [('answers',m+'.json'),('results',m+'-61.json'),('results',m+'-125.json')]:
  link=OUT/folder/name;target=V1/folder/name
  if not link.is_symlink() and not link.exists():link.symlink_to(target)
  references[str(link)]=str(target)
for m in ['none','oracle']:
 link=OUT/'answers'/(m+'.json');target=V3/'answers/inmind'/(m+'.json')
 if not link.is_symlink() and not link.exists():link.symlink_to(target)
 references[str(link)]=str(target)
(OUT/'unchanged-input-references.json').write_text(json.dumps({'reason':'Only raw-method query-feature batching is controlled; native method and calibration inputs/generation are unchanged. References, not new runs.','files':references},indent=2))
queue=['full','hyde','rerank','plain','write_only_attr','fact_presence'];running={};done=[];failed=[];(OUT/'logs').mkdir(exist_ok=True)
while (queue or running) and time.time()<deadline:
 while queue and len(running)<3:
  m=queue.pop(0);f=(OUT/'logs'/(m+'.log')).open('a');env=dict(os.environ,NATIVE_JOB='frozen241-'+m,RUN_JOB='native-v3-frozen241-'+m)
  running[m]=subprocess.Popen([str(ROOT/'.venv-baselines/bin/python'),str(OUT/'run_raw.py'),m],env=env,stdout=f,stderr=subprocess.STDOUT);f.close()
 for m,p in list(running.items()):
  if p.poll() is not None:
   (done if p.returncode==0 else failed).append(m);del running[m]
 tmp=OUT/'status.tmp';tmp.write_text(json.dumps({'updated':time.strftime('%Y-%m-%d %H:%M:%S'),'running':{m:p.pid for m,p in running.items()},'remaining':queue,'completed':done,'failed':failed,'deadline':deadline},indent=2));tmp.replace(OUT/'status.json');time.sleep(5)
