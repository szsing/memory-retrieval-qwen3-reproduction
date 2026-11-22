"""Parallel independent task states; never parallelize updates within a task."""
from common import *
import argparse, subprocess
ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['mem0','amem'],required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
processes=[];files=[]
for i in range(a.workers):
 log=(HERE/(a.method+'-shard-'+str(i)+'.log')).open('a');files.append(log)
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 cmd=[str(ROOT/'scripts/bootstrap.sh'),str(ROOT/'.venv-baselines/bin/python'),'-u',str(HERE/'run_baselines.py'),'--method',a.method,'--shard-index',str(i),'--shard-count',str(a.workers)]
 processes.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT))
dump(HERE/(a.method+'-workers.json'),{'workers':[{'shard':i,'pid':p.pid} for i,p in enumerate(processes)],'note':'disjoint task IDs, separate memory databases, one ordered writer per task'})
codes=[p.wait() for p in processes]
for log in files:log.close()
dump(HERE/(a.method+'-worker-exits.json'),codes)
if any(c!=0 for c in codes):raise RuntimeError('Baseline shard failure: '+str(codes))
tasks,_,_=load_data();rows=[]
for t in tasks:
 p=HERE/'baselines'/a.method/str(t['task_id'])/'complete.json'
 assert p.exists(), 'missing task '+str(t['task_id'])
 r=json.loads(p.read_text());assert r['task_id']==t['task_id'] and r['state_query_isolation'];rows.append(r)
assert len(rows)==125 and len({r['task_id'] for r in rows})==125
dump(HERE/'baselines'/(a.method+'-all125.json'),rows);event('baseline_full_complete',method=a.method,tasks=125)
