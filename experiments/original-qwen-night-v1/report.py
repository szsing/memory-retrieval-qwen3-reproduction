"""Compact coverage/diagnostic report; no partial result is labelled full."""
import os,json,time,argparse,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent

def report():
 state=json.loads((HERE/'status.json').read_text()) if (HERE/'status.json').exists() else {}
 rows=[]
 for f in sorted((HERE/'results').glob('*.json')):
  x=json.loads(f.read_text());res=x['result']
  if x['spec'].get('answer'):
   for bed,r in res.get('res',{}).items():rows.append({'job':f.stem,'metric':bed+'/answer','correct':r['applied'],'n':r['n']})
  elif x['spec']['method'].startswith('legacy_'):
   for bed,rr in res.items():
    for metric,v in rr.items():
     if metric!='task_ids':rows.append({'job':f.stem,'metric':bed+'/'+metric,'correct':int(sum(v)),'n':len(v)})
  else:
   for metric,v in res.items():rows.append({'job':f.stem,'metric':metric,'correct':int(sum(v)),'n':len(v)})
 counts=collections.Counter();jobs=collections.defaultdict(collections.Counter)
 if (HERE/'events.jsonl').exists():
  for line in (HERE/'events.jsonl').open():
   try:r=json.loads(line)
   except json.JSONDecodeError:continue
   if r.get('job','').startswith('preflight'):continue
   c=jobs[r.get('job','unknown')];k=r['kind']
   if k=='api_response':c['api_calls']+=1;c['truncated']+=r.get('finish_reason')=='length';c['thinking']+=bool(r.get('thinking_text') or r.get('reasoning_content'))
   if k=='json_text_check':c['json_attempts']+=1;c['invalid_json']+=not r['valid_json']
   if k=='jcall_return':c['jcall_calls']+=1;c['jcall_none']+=r['returned_none']
   if k=='by_id':c['expected_items']+=r['expected'];c['missing_items']+=r['missing'];c['valid_empty_lists']+=r['empty']
 for c in jobs.values():counts.update(c)
 gate=json.loads((HERE/'alignment-gate.json').read_text()) if (HERE/'alignment-gate.json').exists() else {'status':'pending'}
 obj={'updated':time.strftime('%Y-%m-%d %H:%M:%S'),'state':state,'metrics':rows,'diagnostics':dict(counts),'diagnostics_by_job':{k:dict(v) for k,v in jobs.items()},'full_expansion_gate':gate,'coverage_warning':'Full methods/benchmarks are incomplete unless every fold and comparator finished; compare common folds only.'}
 (HERE/'summary.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2))
 lines=['# 原作者脚本＋Qwen3-8B 夜间进度','', '状态：'+state.get('phase','preflight')+'；当前任务：'+str(state.get('current_job')),'', '原作者源码与数据校验、原提示/批处理/排序规则均保留。以下仅列已完成任务，不把未完成全量当作全量成绩。','', '|任务|指标|命中或应用成功|','|---|---|---:|']
 for r in rows:lines.append('|'+r['job']+'|'+r['metric']+'|'+str(r['correct'])+'/'+str(r['n'])+'|')
 lines+=['','解析/截断汇总：`'+json.dumps(dict(counts),ensure_ascii=False)+'`','', '自动全量决策：`'+str(gate.get('expand_to_full','尚未完成原子集'))+'`。失败率/截断可能导致不触发；完整原因见alignment-gate.json。','', '本轮全量指原作者协议下扩大记录/题目覆盖，不等于官方完整长历史协议。']
 (HERE/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
 return obj
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',type=int);a=p.parse_args()
 proc=Path('/proc')/str(a.parent)/'stat' if a.parent else None;start=proc.read_text().split()[21] if proc and proc.exists() else None
 while True:
  report()
  if not proc or not proc.exists():break
  ps=proc.read_text().split()
  if ps[21]!=start or ps[2]=='Z':break
  time.sleep(30)
 report()
