import json,time,os,argparse,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent
def report():
 state=json.loads((HERE/'status.json').read_text());counts=collections.Counter();latest={}
 for line in (HERE/'events.jsonl').open():
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  if r['kind']=='api_response':counts['calls']+=1;counts['truncated']+=r['finish']=='length';counts['thinking']+=r['thinking']
  if r['kind']=='attempt_failed':counts['rejected_attempts']+=1
  if r['kind']=='job_failed':counts['failed_jobs']+=1
  if r['kind'] in ['ingest','retrieved','evaluated','judged']:latest[r.get('job')]=r
 rows=[json.loads(p.read_text()) for p in sorted((HERE/'results').glob('*.json'))]
 obj={'updated':time.strftime('%Y-%m-%d %H:%M:%S'),'state':state,'diagnostics':dict(counts),'latest_progress':latest,'complete_results':rows,'metric_warning':'Common reader hidden-fact application, not raw Hit@6. InMind is365-line shared-pool protocol;125 includes development questions.'}
 from transport import dump
 dump(HERE/'summary.json',obj)
 text=['# 原生记忆系统同模型对照进度','',obj['updated'],'',f"阶段：{state['phase']}；当前：{state['current']}",'', '只列完整评测；失败不是0分。InMind使用365条共享池；125题含开发题，61题单列。主指标为相同Qwen回答器的隐藏事实利用率，不能与原始行Hit@6混用。','', '|数据|方法|已完成题数|利用隐藏事实|','|---|---|---:|---:|']
 for r in rows:text.append(f"|{r['bed']}|{r['method']}|{r['n']}|{r['correct']}/{r['n']}|")
 text+=['','解析记录：`'+json.dumps(dict(counts))+'`','', '当前检查和建库不是完整复现结论；以checks/、原生状态、逐题上下文及完整结果共同判断。']
 (HERE/'REPORT.zh.md').write_text('\n'.join(text)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',type=int);args=p.parse_args()
 while True:
  report()
  if not args.parent or not (Path('/proc')/str(args.parent)).exists():break
  time.sleep(30)
 report()
