import os,json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1];os.environ['NATIVE_DEADLINE']=(p/'deadline.txt').read_text().strip()
for spec in [{'stage':'retrieve','bed':'inmind','method':'none'},{'stage':'evaluate','bed':'inmind','method':'none','n':61},{'stage':'evaluate','bed':'inmind','method':'none','n':125}]:
 name='-'.join(str(spec.get(k,'')) for k in ['stage','bed','method','n'])
 with (p/'logs'/('control-'+name+'.log')).open('a') as f:
  code=subprocess.run([str(root/'.venv-baselines/bin/python'),str(p/'worker.py'),json.dumps(spec)],env=dict(os.environ,NATIVE_JOB='control-'+name,RUN_JOB='native-v3-control-'+name),stdout=f,stderr=subprocess.STDOUT).returncode
 if code:raise SystemExit(code)
 (p/'done'/(name+'.json')).write_text(json.dumps({'spec':spec,'finished':time.strftime('%Y-%m-%d %H:%M:%S'),'diagnostic_only':True}))
