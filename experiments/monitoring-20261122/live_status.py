import json,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'experiments';V3=E/'native-source-aligned-v3';OUT=E/'other-native-readout4096-v1'
def load(p,default=None):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default
last={};fail=[]
for line in (V3/'events.jsonl').open():
 try:r=json.loads(line)
 except:continue
 if not str(r.get('job','')).startswith('continuation-'):continue
 if r['kind'] in ['ingest','retrieved','job_complete','job_failed']:last[r['job']]=r
 if r['kind'] in ['job_failed','attempt_failed']:fail.append(r)
s={'time':time.strftime('%F %T'),'controller':load(E/'monitoring-20261122/continuation-status.json'),'latest_native_events':last,'recent_errors':fail[-5:],'coverage':{},'readout_errors':{p.stem:load(p) for p in (OUT/'errors').glob('*.json')}}
for b in ['implex','lucid']:
 s['coverage'][b]={}
 for m in ['full','mem0','amem','rootmem','hyde','rerank','plain']:
  c=load(V3/'contexts'/b/(m+'.json'),{});a=load(OUT/'answers'/b/(m+'.json'),{})
  s['coverage'][b][m]={'retrieved':len(c),'answered':len(a),'judged_diagnostic':sum(x.get('applied') is not None for x in a.values())}
(E/'monitoring-20261122/continuation-live.json').write_text(json.dumps(s,ensure_ascii=False,indent=2));(Path(__file__).resolve().parent/'live-status.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
compact={'time':s['time'],'phase':s['controller']['phase'],'coverage':{b:{m:[x['retrieved'],x['answered'],x['judged_diagnostic']] for m,x in v.items() if x['retrieved'] or x['answered']} for b,v in s['coverage'].items()},'building':{k:[v.get('done'),v.get('total')] for k,v in last.items() if v['kind']=='ingest'},'errors':s['recent_errors'],'readout_errors':s['readout_errors']}
print(json.dumps(compact,ensure_ascii=False))

