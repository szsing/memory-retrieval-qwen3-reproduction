import os,json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1]
os.environ['NATIVE_DEADLINE']=(p/'deadline.txt').read_text().strip()
for spec in [{'stage':'retrieve','bed':'inmind','method':'full'},{'stage':'evaluate','bed':'inmind','method':'full','n':61}]:
 name='-'.join(str(spec.get(k,'')) for k in ['stage','bed','method','n'])
 done=p/'done'/(name+'.json')
 if done.exists():continue
 env=dict(os.environ,NATIVE_JOB=name,RUN_JOB='native-v3-'+name)
 with (p/'logs'/('priority-'+name+'.log')).open('a') as f:
  code=subprocess.run([str(root/'.venv-baselines/bin/python'),str(p/'worker.py'),json.dumps(spec)],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 if code:raise SystemExit(code)
 done.parent.mkdir(exist_ok=True)
 done.write_text(json.dumps({'spec':spec,'finished':time.strftime('%Y-%m-%d %H:%M:%S'),'priority_comparison':True}))
