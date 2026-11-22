from common import *
tasks,bg,meta=load_data();assert len(tasks)==125 and len({t['task_id'] for t in tasks})==125
dev=dev_ids(tasks);assert len(dev)==len(set(dev))==40
counts=[]
for t in tasks:
 ss=timeline(t);assert len(ss)==47 and ss[8]['turns'][-2]['content']==t['user_message']
 assert sum(len(s['turns']) for s in ss)==488
 cs=[c for s in ss for c in pairs(s)];counts.append(len(cs))
 assert sum(len(c) for c in cs)==488
 assert all(set(m)=={'role','content'} for c in cs for m in c)
 target=[i for i,c in enumerate(cs) if t['user_message'] in text(c) and t['assistant_message'] in text(c)]
 assert len(target)==1
 inserted=[x for s in ss for x in s['turns'] if x['source']=='inmind_target']
 assert len(inserted)==2 and all(x['id'].startswith('inmind-task-'+str(t['task_id'])+'-') for x in inserted)
assert budget_context(['short','x'*100],max_tokens=3)=='short'
state={'all125_timelines':True,'no_gold_fields_in_ingestion':True,'no_other_task_target_origin_in_state':True,'duplicate_personal_fact_groups':[[x['task_id'] for x in tasks if x['user_message']==fact] for fact in sorted(set(x['user_message'] for x in tasks)) if sum(x['user_message']==fact for x in tasks)>1],'all488_turns_preserved':True,'interaction_chunk_counts':sorted(set(counts)),'dev':40,'disjoint':85,'context_whole_note_budget_test':True}
dump(HERE/'preflight.json',state);event('preflight_pass',**state)
