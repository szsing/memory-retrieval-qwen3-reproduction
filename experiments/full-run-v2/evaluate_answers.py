"""Local end-to-end adaptation using original InMind answer/judge prompts, all 125 tasks."""
from common import *
OUT=HERE/'answers';OUT.mkdir(exist_ok=True);T=Transport('answers-and-judge')
tasks,_,_=load_data();byid={t['task_id']:t for t in tasks};dev=set(dev_ids(tasks))
recipe=json.loads((HERE/'retrieval/recipe.lock.json').read_text());chosen=recipe['selected']
rows=json.loads((HERE/'retrieval/all125-results.json').read_text())
sources={name:{(r['task_id'],r['kind']):r['context'] for r in rows if r['method']==name} for name in ['dense','hyde',chosen]}
control=json.loads((HERE/'retrieval/dense-pick-control.json').read_text())
assert len(control)==250
sources['dense_pick']={(r['task_id'],r['kind']):r['context'] for r in control}
for name in ['mem0','amem']:
 records=json.loads((HERE/'baselines'/(name+'-all125.json')).read_text())
 assert len(records)==125 and len({r['task_id'] for r in records})==125
 sources[name]={(r['task_id'],x['kind']):x['context'] for r in records for x in r['results']}
prompts=VENDOR/'InMind-main/evaluation/prompts'
schema={'type':'object','properties':{'score':{'type':'integer','enum':[0,1]},'reason':{'type':'string'}},'required':['score','reason']}
for name,contexts in sources.items():
 completed=[]
 for task in tasks:
  p=OUT/name/(str(task['task_id'])+'.json');p.parent.mkdir(exist_ok=True)
  if p.exists():completed.append(json.loads(p.read_text()));continue
  row={'task_id':task['task_id'],'system':name,'config':{'answer_model':MODEL,'judge_model':MODEL,'retrieval_top_k':6,'context_tokens':4096,'protocol':'local controlled adaptation, not official-model leaderboard replication'}}
  for kind,outputkey in [('query','query'),('naive_query','naive')]:
   ctx=contexts[(task['task_id'],kind)];system=(prompts/'answer_system.txt').read_text().replace('{context}',ctx)
   answer=T.call([{'role':'system','content':system},{'role':'user','content':task[kind]}])
   payload={k:task[k] for k in ['user_message','explanation']};payload.update(context=ctx,query=task[kind],answer=answer)
   metric='judge_application.txt' if kind=='query' else 'judge_naive.txt'
   judged=T.call([{'role':'system','content':(prompts/metric).read_text()},{'role':'user','content':json.dumps(payload)}],schema=schema,json_output=True)
   payload.pop('answer')
   recall=T.call([{'role':'system','content':(prompts/'judge_target_recall.txt').read_text()},{'role':'user','content':json.dumps(payload)}],schema=schema,json_output=True)
   row[outputkey]={'context':ctx,'answer':answer,'application_or_naive':judged,'target_recall':recall}
  dump(p,row);completed.append(row);event('answer_task',method=name,done=len(completed),total=125)
 out=OUT/(name+'-submission.jsonl');out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in completed))
 import subprocess
 subprocess.run([sys.executable,str(VENDOR/'InMind-main/evaluation/scripts/validate_submission.py'),str(out)],check=True)
 summaries={}
 for group,rr in [('all125',completed),('disjoint85',[r for r in completed if r['task_id'] not in dev]),('cross_domain61',[r for r in completed if byid[r['task_id']]['domain'] not in {'health_and_wellness','legal','consumer','personal_development','other'}])]:
  summaries[group]={'n':len(rr),'application':sum(r['query']['application_or_naive']['score'] for r in rr),'target_recall':sum(r['query']['target_recall']['score'] for r in rr),'naive':sum(r['naive']['application_or_naive']['score'] for r in rr)}
 dump(OUT/(name+'-summary.json'),summaries)
dump(OUT/'complete.json',{'methods':list(sources),'tasks':125,'paid_api_calls':0,'judge_limitation':'same local backbone; requires later independent-model/human agreement checks'})
