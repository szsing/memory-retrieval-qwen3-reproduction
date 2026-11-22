import os,json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1]
os.environ['NATIVE_DEADLINE']=(p/'deadline.txt').read_text().strip()
jobs=[{'stage':'retrieve','bed':'inmind','method':'rootmem'},{'stage':'evaluate','bed':'inmind','method':'rootmem','n':61},{'stage':'evaluate','bed':'inmind','method':'rootmem','n':125},{'stage':'retrieve','bed':'implex','method':'mem0'},{'stage':'evaluate','bed':'implex','method':'mem0','n':81},{'stage':'retrieve','bed':'implex','method':'amem'},{'stage':'evaluate','bed':'implex','method':'amem','n':81}]
state={'started':time.strftime('%Y-%m-%d %H:%M:%S'),'completed':[],'failed':[]}
for spec in jobs:
 if time.time()>=float(os.environ['NATIVE_DEADLINE']):state['phase']='budget';break
 name='-'.join(str(spec.get(k,'')) for k in ['stage','bed','method','n']);done=p/'done'/(name+'.json')
 if done.exists():continue
 if spec['stage']=='evaluate' and not (p/'contexts'/spec['bed']/(spec['method']+'.json')).exists():continue
 state['current']=spec;(p/'repair-status.json').write_text(json.dumps(state,indent=2))
 env=dict(os.environ,NATIVE_JOB='repair-'+name,RUN_JOB='native-v3-repair-'+name)
 with (p/'logs'/('repair-'+name+'.log')).open('a') as f:code=subprocess.run([str(root/'.venv-baselines/bin/python'),str(p/'worker.py'),json.dumps(spec)],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 if code:state['failed'].append({'spec':spec,'code':code})
 else:
  done.write_text(json.dumps({'spec':spec,'repaired':True,'finished':time.strftime('%Y-%m-%d %H:%M:%S')}));state['completed'].append(name)
 (p/'repair-status.json').write_text(json.dumps(state,indent=2))
state['ended']=time.strftime('%Y-%m-%d %H:%M:%S');(p/'repair-status.json').write_text(json.dumps(state,indent=2))
