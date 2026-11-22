"""Recompute published tables without models, private inputs, or network calls."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
METHODS=['full','write_only_attr','fact_presence','mem0','amem','rootmem','hyde','rerank','plain','none','oracle']
def rows(p):
 assert p.exists(),f'Missing {p.relative_to(R)}'
 return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def load(p):return json.loads(p.read_text())
S=load(R/'results/eval-official-v1/summary.json')['results'];computed={};checked=0
for m in METHODS:
 P=rows(R/'results/native-readout4096-v1/predictions'/(m+'.jsonl'));P={x['index']:x for x in P};assert set(P)==set(range(125)),m
 values={}
 for metric in ['answer','grounded']:
  J=rows(R/'results/eval-official-v1/rows/native_readout4096'/metric/(m+'.jsonl'));assert len(J)==125
  assert {x['i'] for x in J}==set(range(125));assert all(x['score'] in [0,1] for x in J)
  for j in J:
   p=P[j['i']];assert digest(p['answer'])==j['answer_sha256'],(m,j['i'],'answer mismatch')
   assert p['context_text_sha256']==j['notes_sha256'],(m,j['i'],'context digest mismatch')
   assert str(p['id'])==str(j['task_id']),(m,j['i'],'task mismatch');checked+=1
  v61=sum(x['score'] for x in J if x['i']<61);v125=sum(x['score'] for x in J);v=S['native_readout4096/'+metric+'/'+m]
  assert (v61,v125)==(v['correct61'],v['correct125']),(m,metric,'aggregate mismatch')
  values[metric]=(v61,v125)
 computed[m]=values
F=load(R/'results/strict241-queryfrozen-v2/frozen-query-outputs.json');I=load(R/'manifests/task-and-pool-identity.json')['inmind']['tasks']
assert F['query_sha256']==[t['query_sha256'] for t in I]
assert all(len(F[k])==125 for k in ['hyde','properties','gate'])
print('method\tanswer61\tanswer125\tgrounded61\tgrounded125')
for m,v in computed.items():print(m,*v['answer'],*v['grounded'],sep='\t')
assert checked==2750;print(f'PASS: {checked} primary judge rows; answer hashes, context digest agreement, task IDs and aggregates match.')
labels=['full','mem0','amem','rootmem','hyde','rerank','plain']
for n in [10,20]:
 C=load(R/f'results/monitoring-20261122/common-prefix{n}-results.json')['rows']
 for m in labels:
  P={x['index']:x for x in rows(R/'results/other-native-readout4096-v1/predictions/implex'/(m+'.jsonl'))}
  assert all(i in P and P[i]['applied'] is not None for i in range(n))
  v=sum(P[i]['applied'] for i in range(n));assert C[m]['n']==n and C[m]['source_diagnostic_correct']==v
 print(f'PASS: matched first {n} ImplexConv diagnostic rows for all seven methods.')
print('LIMIT: source contexts are not republished; context digests are cross-checked, not recomputed here from hidden input text.')
print('LIMIT: these checks do not validate LLM-judge correctness or establish method effectiveness.')
