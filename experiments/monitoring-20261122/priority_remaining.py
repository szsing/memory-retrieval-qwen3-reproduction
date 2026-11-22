"""Run short remaining raw comparisons then RootMem; preserve paused native construction."""
import os,json,time,signal,subprocess,traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];E=ROOT/'experiments';V3=E/'native-source-aligned-v3';R=E/'other-native-readout4096-v1';deadline=float((V3/'deadline.txt').read_text())
os.environ.update(NATIVE_DEADLINE=str(deadline),NIGHT_DEADLINE_UNIX=str(deadline));state={'deadline':deadline,'paused_parent':63041,'paused_child':68554,'completed':[],'errors':[],'phase':'priority_raw_then_rootmem'}
def save():
 p=OUT/'priority-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
def resume(pid,needle):
 try:
  cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if needle in cmd:os.kill(pid,signal.SIGCONT)
 except (FileNotFoundError,ProcessLookupError):pass
try:
 for method in ['hyde','rerank','plain','rootmem']:
  if time.time()>=deadline:break
  state['current']=method;save()
  for stage in ['retrieve','evaluate']:
   if time.time()>=deadline:break
   if (R/'results'/f'lucid-{method}-240.json').exists():break
   c=V3/'contexts/lucid'/(method+'.json')
   if stage=='retrieve' and c.exists() and len(json.loads(c.read_text()))==240:continue
   if stage=='evaluate' and (not c.exists() or len(json.loads(c.read_text()))!=240):break
   spec={'stage':stage,'bed':'lucid','method':method,'n':240};env=dict(os.environ,NATIVE_JOB='continuation-priority-lucid-'+method+'-'+stage,RUN_JOB='native-v3-priority-lucid-'+method+'-'+stage)
   with (V3/'logs'/('priority-lucid-'+method+'-'+stage+'.log')).open('a') as f:
    code=subprocess.run([str(ROOT/'.venv-baselines/bin/python'),str(R/('retrieve.py' if stage=='retrieve' else 'evaluate.py')),json.dumps(spec)],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
   if code:state['errors'].append({'method':method,'stage':stage,'code':code});save();break
  if (R/'results'/f'lucid-{method}-240.json').exists():state['completed'].append(method);save()
except BaseException as e:
 state['exception']=repr(e);state['traceback']=traceback.format_exc();raise
finally:
 resume(68554,'other-native-readout4096-v1/retrieve.py');resume(63041,'monitoring-20261122/continue_after_primary.py');state['phase']='finished_or_budget';state['ended']=time.strftime('%F %T');save()

