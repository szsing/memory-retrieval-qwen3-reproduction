"""Real full-case protocol checks before durable night queue launch."""
from night_data import *
from night_worker import load_case,run_case
from night_methods import orders
from night_eval import evaluate

def main():
 report={'passed':False,'cases':[],'gold_input_isolation':True}
 for ds in ['inmind','lucid_s','implex']:
  with connect() as c:cid=c.execute("select id from cases where dataset=? and queries!='[]' order by rowid limit 1",(ds,)).fetchone()[0]
  notes,qs=load_case(ds,cid);out=orders(notes,qs,'dense')
  assert len(out)==len(qs)
  for n in notes:assert all(set(m)=={'role','content'} for m in json.loads(n['messages']))
  q=qs[0];top=out[0]['dense'][:6];context=budget_context([notes[i]['text'] for i in top]);assert token_count(context)<=4096
  # Perturb gold metadata and verify pure retrieval sees only text, no labels.
  changed=[{**q,'gold':{'deliberately':'invalid'},'gold_sessions':['nonexistent']} for q in qs]
  assert orders(notes,changed,'dense')==out
  evaluated=evaluate(ds,q,context)
  assert evaluated['score'] in (0,1)
  report['cases'].append({'dataset':ds,'id':cid,'chunks':len(notes),'questions':len(qs),'reader_and_judge_passed':True,'score_not_a_claim':evaluated['score']})
 # InMind exact source target Hit@6 must match the frozen completed run on every task.
 tasks,_,_=load_data();expected=json.loads((HERE/'retrieval/all125-results.json').read_text());dense={(str(r['task_id']),r['kind']):r['hit_at_6'] for r in expected if r['method']=='dense'}
 correct=0
 for t in tasks:
  notes,qs=load_case('inmind',str(t['task_id']));oo=orders(notes,qs,'dense')
  for q,o in zip(qs,oo):
   hit=int(any(t['user_message'] in notes[i]['text'] and t['assistant_message'] in notes[i]['text'] for i in o['dense'][:6]))
   assert hit==dense[str(t['task_id']),q['id']],('baseline mismatch',t['task_id'],q['id']);correct+=1
 report['inmind_dense_consistency_queries']=correct
 report['passed']=True;dump(HERE/'night-preflight.json',report);event('night_preflight_passed',**report)
if __name__=='__main__':main()
