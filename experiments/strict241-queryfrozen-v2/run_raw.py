"""Freeze query-only outputs at the original global125 batching; isolate only history pools."""
import os,sys,json,hashlib,time,shutil,fcntl
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];V1=ROOT/'experiments/strict241-v1';V3=ROOT/'experiments/native-source-aligned-v3'
sys.path.insert(0,str(V1));import run_method as S
import methods as M
NAME=sys.argv[1];assert NAME in ['full','hyde','rerank','plain','write_only_attr','fact_presence']
from transport import dump
OUT.mkdir(exist_ok=True);qs=[t['q'] for t in S.TASKS];assert len(set(qs))==125
with (OUT/'feature.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);feature=OUT/'query-features.json'
 if not feature.exists():
  before=dict(S.W.lm.STATS);S.W.bridge.PV[0]=2
  data={'queries':qs,'hyde':M.gen_hyde(qs),'properties':S.W.bridge.props_reqs(qs),'gate':S.W.bridge.gate_reqs(qs),'batch_sizes':{'hyde':15,'properties':8,'gate':20},'model':S.W.lm.MODEL,'new_api_calls':S.W.lm.STATS['calls']-before['calls'],'validated_cache_hits':S.W.lm.STATS['cached']-before['cached']}
  assert all(len(data[k])==125 for k in ['hyde','properties','gate']);dump(feature,data)
  # A read-only vector-cache snapshot prevents parallel runners from overwriting one cache file.
  import numpy as np
  target=OUT/'embedding-cache';target.mkdir(exist_ok=True)
  source=Path(S.W.ret._path())
  if source.exists():
   for attempt in range(5):
    try:
     with np.load(source,allow_pickle=False) as z:k=z['k'];v=z['v']
     np.savez(target/'emb.npz',k=k,v=v);break
    except Exception:
     if attempt==4:raise
     time.sleep(.2)
data=json.loads(feature.read_text());assert data['queries']==qs;lookup={q:i for i,q in enumerate(qs)}
def lookup_values(which,queries):return [data[which][lookup[q]] for q in queries]
M.gen_hyde=lambda queries:lookup_values('hyde',queries)
S.W.bridge.props_reqs=lambda queries:lookup_values('properties',queries)
S.W.bridge.gate_reqs=lambda queries:lookup_values('gate',queries)
S.W.ret.CDIR=str(OUT/'embedding-cache');S.W.ret.save=lambda:None
S.METHOD='full' if NAME in ['write_only_attr','fact_presence'] else NAME
S.EXP=OUT/'variants'/NAME if NAME in ['write_only_attr','fact_presence'] else OUT
if NAME=='write_only_attr':S.W.bridge.props_reqs=lambda queries:[[q] for q in queries]
if NAME=='fact_presence':
 import numpy as np
 original_scores=S.W.bench.scores
 def presence_scores(lines,queries,parts,pools=None):
  baseline=original_scores(lines,queries,{'hyde'},pools)
  def presence(self,rp):
   has=np.zeros(self.n);has[self.own]=1.
   return [(has if props else 0.)+1e-3*s for props,s in zip(rp,baseline)]
  S.W.bridge.PropIndex.bonus=presence
  return original_scores(lines,queries,parts,pools)
 S.W.bench.scores=presence_scores
save_original=S.save_result
def publish(rows,n):
 save_original(rows,n);f=S.EXP/'results'/f'{S.METHOD}-{n}.json'
 if f.exists():
  result=json.loads(f.read_text());result.update(method=NAME,query_features_frozen=True,query_feature_sha256=hashlib.sha256(feature.read_bytes()).hexdigest(),write_side_pool_isolated=True)
  dump(OUT/'results'/f'{NAME}-{n}.json',result)
S.save_result=publish
if NAME in ['write_only_attr','fact_presence']:
 (OUT/'answers').mkdir(exist_ok=True);link=OUT/'answers'/(NAME+'.json')
 if not link.exists() and not link.is_symlink():link.symlink_to(S.EXP/'answers/full.json')
try:S.main()
except BaseException as e:
 import traceback
 dump(OUT/'errors'/(NAME+'.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
