"""Unmodified official InMind judge prompts over frozen answers; local Qwen adaptation."""
import os,sys,json,time,threading,hashlib,concurrent.futures,traceback
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1];V3=ROOT/'experiments/native-source-aligned-v3';STRICT=ROOT/'experiments/strict241-v1'
FROZEN=ROOT/'experiments/strict241-queryfrozen-v2'
PROTOCOLS=[('native_readout4096',ROOT/'experiments/native-readout4096-v1'),('isolated241_frozen_queries',FROZEN),('shared365',V3),('isolated241',STRICT)]
PROTOCOLS.append(('root_native4096_sensitivity',ROOT/'experiments/root-native4096-v1'))
sys.path.insert(0,str(V3));from transport import Transport,dump,budget
os.environ['NATIVE_DEADLINE']=(V3/'deadline.txt').read_text().strip();os.environ['NATIVE_JOB']='official-prompt-rescore'
TASKS=json.loads((V3/'datasets.json').read_text())['inmind']['tasks'];SOURCE={str(t['task_id']):t for t in map(json.loads,(ROOT/'rsi-memory/inmind_data/benchmark/dataset/inmind.jsonl').open())}
PROMPT_DIR=ROOT/'experiments/full-run-v2/vendor/InMind-main/evaluation/prompts'
PROMPTS={k:(PROMPT_DIR/filename).read_text() for k,filename in [('answer','judge_answer_only.txt'),('grounded','judge_application.txt')]}
SCHEMA={'type':'object','properties':{'score':{'type':'integer','enum':[0,1]},'reason':{'type':'string'}},'required':['score','reason'],'additionalProperties':False}
METHODS=['full','rootmem','mem0','amem','hyde','rerank','plain','none','oracle','write_only_attr','fact_presence'];lock=threading.Lock();local=threading.local()
def score(protocol,method,i,metric,row):
 if not hasattr(local,'transport'):local.transport=Transport()
 gold=SOURCE[str(TASKS[i]['id'])];assert gold['user_message']==TASKS[i]['fact']
 payload={'user_message':gold['user_message'],'query':gold['query'],'explanation':gold['explanation'],'answer':row['answer']}
 if metric=='grounded':payload['context']=row['notes']
 result=local.transport.call([{'role':'system','content':PROMPTS[metric]},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],schema=SCHEMA,json_output=True)
 result.update(i=i,task_id=gold['task_id'],method=method,protocol=protocol,metric=metric,answer_sha256=hashlib.sha256(row['answer'].encode()).hexdigest(),notes_sha256=hashlib.sha256(row['notes'].encode()).hexdigest())
 result['empty_context_invariant_failed']=metric=='grounded' and method=='none' and result['score']!=0
 dump(EXP/'rows'/protocol/metric/method/(str(i)+'.json'),result);return result
def summarize():
 results={};invariant=[]
 for protocol,_ in PROTOCOLS:
  for metric in PROMPTS:
   for method in METHODS:
    path=EXP/'rows'/protocol/metric/method;rows={int(p.stem):json.loads(p.read_text()) for p in path.glob('*.json')}
    if not rows:continue
    key='/'.join([protocol,metric,method]);v={'evaluated':len(rows),'correct_evaluated':sum(x['score'] for x in rows.values()),'heldout61_complete':all(i in rows for i in range(61)),'all125_complete':len(rows)==125}
    if v['heldout61_complete']:v['correct61']=sum(rows[i]['score'] for i in range(61))
    if v['all125_complete']:v['correct125']=sum(rows[i]['score'] for i in range(125))
    results[key]=v;invariant +=[r for r in rows.values() if r['empty_context_invariant_failed']]
 dump(EXP/'summary.json',{'updated':time.strftime('%Y-%m-%d %H:%M:%S'),'source':'official judge_answer_only.txt and judge_application.txt; same local Qwen backend; reader/generation not changed','results':results,'empty_context_invariant_failures':invariant})
def main():
 EXP.mkdir(exist_ok=True);dump(EXP/'protocol.json',{'model':'Qwen3-8B same local endpoint','temperature':0,'max_tokens':2048,'prompts':{k:{'path':str(PROMPT_DIR/('judge_answer_only.txt' if k=='answer' else 'judge_application.txt')),'sha256':hashlib.sha256(v.encode()).hexdigest()} for k,v in PROMPTS.items()},'primary_old_scores_unchanged':True,'answer_generation_unchanged':True,'source_explanations_only_used_in_judging':True})
 # First score the prespecified blinded calibration sample before reporting full rankings.
 sample=[x['i'] for x in json.loads((V3/'blind-review.json').read_text())['cases']]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for metric in ['answer','grounded']:
   tasks=[]
   for method in ['none','full','rootmem']:
    answers=json.loads((V3/'answers/inmind'/(method+'.json')).read_text())
    for i in sample:
     if not (EXP/'rows/shared365'/metric/method/(str(i)+'.json')).exists():tasks.append(pool.submit(score,'shared365',method,i,metric,answers[str(i)]))
   for future in concurrent.futures.as_completed(tasks):future.result()
  dump(EXP/'calibration-complete.json',{'n_queries':8,'n_answers':24,'n_metrics':2,'completed':time.strftime('%H:%M:%S')});summarize()
  while time.time()<float(os.environ['NATIVE_DEADLINE']):
   pending=[]
   # Prioritize frozen61 heldout questions across methods before the descriptive125 expansion.
   for limit in [61,125]:
    for protocol,base in PROTOCOLS:
     for metric in ['answer','grounded']:
      for method in METHODS:
       f=base/('answers/inmind' if protocol=='shared365' else 'answers')/(method+'.json')
       if not f.exists():continue
       answers=json.loads(f.read_text())
       for i in range(limit):
        if str(i) not in answers:continue
        if not (EXP/'rows'/protocol/metric/method/(str(i)+'.json')).exists():pending.append((protocol,method,i,metric,answers[str(i)]))
    if pending:break
   if not pending:summarize();time.sleep(10);continue
   # Refresh available answers and priorities after each small batch.
   pending=pending[:16]
   for start in range(0,len(pending),16):
    budget();futures=[pool.submit(score,*s) for s in pending[start:start+16]]
    for future in concurrent.futures.as_completed(futures):future.result()
    summarize()
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(EXP/'error.json',{'error':repr(e),'traceback':traceback.format_exc()});raise

