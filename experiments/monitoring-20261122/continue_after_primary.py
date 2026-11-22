"""Spend the remaining authorized window on other subsets after the primary verification completes."""
import os,json,time,signal,subprocess,concurrent.futures,shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];V3=ROOT/'experiments/native-source-aligned-v3';PRIMARY=ROOT/'experiments/native-readout4096-v1';EVAL=ROOT/'experiments/eval-official-v1';READOUT=ROOT/'experiments/other-native-readout4096-v1'
deadline=float((V3/'deadline.txt').read_text());os.environ.update(NATIVE_DEADLINE=str(deadline),NIGHT_DEADLINE_UNIX=str(deadline))
methods=['full','rootmem','mem0','amem','hyde','rerank','plain','write_only_attr','fact_presence']
def primary_ready():
 for m in methods:
  f=PRIMARY/'answers'/(m+'.json')
  if not f.exists() or len(json.loads(f.read_text()))!=125:return False
  for metric in ['answer','grounded']:
   if not all((EVAL/'rows/native_readout4096'/metric/m/(str(i)+'.json')).exists() for i in range(125)):return False
 return True
def dump(name,value):
 p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
state={'phase':'waiting_for_primary125_and_official_rescores','deadline':deadline,'completed':[],'failed':[],'active':[]};dump('continuation-status.json',state)
while time.time()<deadline and not primary_ready():time.sleep(10)
if time.time()>=deadline:state['phase']='budget';dump('continuation-status.json',state);raise SystemExit(0)
stopped=[]
for pid in [24573,34920,35245]:
 try:
  cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\x00',b' ').decode()
  if 'native-source-aligned-v3/' in cmd:os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);stopped.append(pid)
 except (FileNotFoundError,ProcessLookupError):pass
cache=READOUT/'embedding-cache-readonly';cache.mkdir(parents=True,exist_ok=True)
shutil.copy2(ROOT/'experiments/original-qwen-format-v2/snapshot/rsi-memory/cache/emb.npz',cache/'emb.npz')
state.update(phase='continuing_other_subsets',replaced_paused_supervisors=stopped);dump('continuation-status.json',state)
def run_pair(bed,method,n):
 label=bed+'-'+method;logdir=V3/'logs';logdir.mkdir(exist_ok=True)
 if (READOUT/'results'/f'{bed}-{method}-{n}.json').exists():return {'label':label,'status':'already_complete'}
 for stage in ['retrieve','evaluate']:
  if time.time()>=deadline:return {'label':label,'status':'budget'}
  c=V3/'contexts'/bed/(method+'.json')
  if stage=='retrieve' and c.exists() and all(str(i) in json.loads(c.read_text()) for i in range(n)):continue
  if stage=='evaluate' and (not c.exists() or not all(str(i) in json.loads(c.read_text()) for i in range(n))):return {'label':label,'status':'incomplete_contexts_not_scored'}
  spec={'stage':stage,'bed':bed,'method':method,'n':n};env=dict(os.environ,NATIVE_JOB='continuation-'+label+'-'+stage,RUN_JOB='native-v3-continuation-'+label+'-'+stage)
  with (logdir/('continuation-'+label+'-'+stage+'.log')).open('a') as f:code=subprocess.run([str(ROOT/'.venv-baselines/bin/python'),str(READOUT/'retrieve.py' if stage=='retrieve' else READOUT/'evaluate.py'),json.dumps(spec)],stdout=f,stderr=subprocess.STDOUT,env=env).returncode
  if code:return {'label':label,'status':'budget_or_error','code':code}
 return {'label':label,'status':'complete'}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 jobs=[(bed,m,n) for bed,n in [('implex',81),('lucid',240)] for m in ['full','mem0','amem','rootmem','hyde','rerank','plain']]
 futures={pool.submit(run_pair,*s):s for s in jobs}
 for f in concurrent.futures.as_completed(futures):
  result=f.result();state['completed' if result['status'] in ['complete','already_complete'] else 'failed'].append(result);dump('continuation-status.json',state)
state['phase']='finished_or_budget';state['ended']=time.strftime('%Y-%m-%d %H:%M:%S');dump('continuation-status.json',state)

