"""Prespecified first full10-item source-judge block; all compared methods use identical queries."""
import os,json,time,subprocess,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;E=O.parent;ROOT=E.parent;R=E/'other-native-readout4096-v1';deadline=(E/'native-source-aligned-v3/deadline.txt').read_text().strip();os.environ.update(NATIVE_DEADLINE=deadline,NIGHT_DEADLINE_UNIX=deadline)
methods=['full','hyde','rerank','plain','rootmem','mem0','amem'];n=20
assert float(deadline)-time.time()>=180,'Less than180s remain; do not launch extension'
manifest={'prespecified':time.strftime('%F %T'),'indices':list(range(n)),'selection':'first two complete10-item source-judge blocks; extend only when both native context prefixes20complete and at least180s remain; not selected by accuracy; diagnostic only','full81_jobs_continue':True,'context_prefix_sha256':{}}
for m in ['mem0','amem']:
 c=json.loads((E/'native-source-aligned-v3/contexts/implex'/(m+'.json')).read_text());assert all(str(i) in c for i in range(n));manifest['context_prefix_sha256'][m]=hashlib.sha256(json.dumps({str(i):c[str(i)] for i in range(n)},sort_keys=True).encode()).hexdigest()
(O/'common-prefix20-manifest.json').write_text(json.dumps(manifest,indent=2))
for m in ['mem0','amem']:
 env=dict(os.environ,NATIVE_JOB='common20-implex-'+m,RUN_JOB='native-v3-common20-'+m)
 with (O/('common20-'+m+'.log')).open('a') as f:
  code=subprocess.run([str(ROOT/'.venv-baselines/bin/python'),str(R/'evaluate.py'),json.dumps({'bed':'implex','method':m,'n':n})],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 if code:raise RuntimeError((m,code))
rows={}
for m in methods:
 a=json.loads((R/'answers/implex'/(m+'.json')).read_text());assert all(str(i) in a and a[str(i)]['applied'] is not None for i in range(n));rows[m]={'n':n,'source_diagnostic_correct':sum(a[str(i)]['applied'] for i in range(n)),'raw_hit6':sum(bool(a[str(i)]['raw_hit6']) for i in range(n)) if m in ['full','hyde','rerank','plain'] else None}
(O/'common-prefix20-results.json').write_text(json.dumps({'completed':time.strftime('%F %T'),'metric':'source simplified judge only; not official accuracy or full81 results','rows':rows},indent=2))

