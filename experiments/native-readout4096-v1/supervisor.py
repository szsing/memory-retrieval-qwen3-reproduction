import os,json,time,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];OLD=ROOT/'experiments/strict241-queryfrozen-v2';V3=ROOT/'experiments/native-source-aligned-v3';deadline=float((V3/'deadline.txt').read_text());os.environ.update(NATIVE_DEADLINE=str(deadline),NIGHT_DEADLINE_UNIX=str(deadline))
(OUT/'answers').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
references={'rootmem':ROOT/'experiments/root-native4096-v1/answers/rootmem.json','none':V3/'answers/inmind/none.json','oracle':V3/'answers/inmind/oracle.json'}
for m,target in references.items():
 link=OUT/'answers'/(m+'.json')
 if not link.exists() and not link.is_symlink():link.symlink_to(target)
(OUT/'references.json').write_text(json.dumps({'reason':'Root4096 generated once using the identical packing rule; none/oracle inputs unchanged when the entire target fact fits the budget. Explicit references, not new generations.','sources':{m:str(p) for m,p in references.items()}},indent=2))
queue=['amem','mem0','full','hyde','rerank','plain','write_only_attr','fact_presence'];running={};done=[];failed=[]
while (queue or running) and time.time()<deadline:
 while queue and len(running)<3:
  m=queue.pop(0);f=(OUT/'logs'/(m+'.log')).open('a');running[m]=subprocess.Popen([str(ROOT/'.venv-baselines/bin/python'),str(OUT/'run_method.py'),m],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy());f.close()
 for m,p in list(running.items()):
  if p.poll() is not None:(done if p.returncode==0 else failed).append(m);del running[m]
 tmp=OUT/'status.tmp';tmp.write_text(json.dumps({'updated':time.strftime('%Y-%m-%d %H:%M:%S'),'running':{m:p.pid for m,p in running.items()},'remaining':queue,'completed':done,'failed':failed,'deadline':deadline},indent=2));tmp.replace(OUT/'status.json');time.sleep(5)
