import os,time,json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1];deadline=float((p/'deadline.txt').read_text());os.environ['NATIVE_DEADLINE']=str(deadline)
while time.time()<deadline and not (p/'results/inmind-rootmem-125.json').exists():time.sleep(5)
for spec in [{'stage':'retrieve','bed':'implex','method':'amem'},{'stage':'evaluate','bed':'implex','method':'amem','n':81}]:
 if time.time()>=deadline:break
 name='-'.join(str(spec.get(k,'')) for k in ['stage','bed','method','n']);done=p/'done'/(name+'.json')
 if done.exists():continue
 with (p/'logs'/('repair-'+name+'.log')).open('a') as f:
  code=subprocess.run([str(root/'.venv-baselines/bin/python'),str(p/'worker.py'),json.dumps(spec)],env=dict(os.environ,NATIVE_JOB='repair-'+name,RUN_JOB='native-v3-repair-'+name),stdout=f,stderr=subprocess.STDOUT).returncode
 if code:raise SystemExit(code)
 done.write_text(json.dumps({'spec':spec,'finished':time.strftime('%Y-%m-%d %H:%M:%S'),'repaired':True}))
