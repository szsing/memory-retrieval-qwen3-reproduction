"""Coverage is always numerator/official denominator, never only successful rows."""
from night_data import *
from collections import defaultdict,Counter

def status():
 report={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'datasets':{},'jobs':{},'comparisons':{},'failures':[]}
 with connect() as c:
  cases=c.execute('select dataset,id,queries from cases').fetchall()
  expected=Counter()
  for ds,cid,qs in cases:expected[ds]+=len(json.loads(qs))
  for ds,n in expected.items():report['datasets'][ds]={'official_questions':n}
  for ds,m,s,n in c.execute('select dataset,method,status,count(*) from jobs group by dataset,method,status'):report['jobs'].setdefault(ds,{}).setdefault(m,{})[s]=n
  for ds,cid,m,e in c.execute("select dataset,id,method,error from jobs where status='failed' limit 20"):report['failures'].append({'dataset':ds,'case':cid,'method':m,'error':e[-1000:]})
  groups=defaultdict(list)
  for ds,cid,m,qid,v in c.execute('select dataset,id,method,qid,value from results'):groups[ds,m].append((cid,qid,json.loads(v)))
  for (ds,m),rows in groups.items():
   ev=[v['evaluation'] for _,_,v in rows if 'evaluation' in v];hits=[v['meta']['gold_session_hit6'] for _,_,v in rows if v['meta'].get('gold_session_hit6') is not None]
   report['datasets'][ds].setdefault('methods',{})[m]={'retrieval_completed':len(rows),'denominator':expected[ds],'evaluated':len(ev),'evaluation_errors':sum('evaluation_error' in v for _,_,v in rows),'correct':sum(v['score'] for v in ev),'accuracy_on_evaluated_only':sum(v['score'] for v in ev)/len(ev) if ev else None,'complete':len(rows)==expected[ds] and len(ev)==expected[ds],'raw_session_hit6':sum(hits)/len(hits) if hits else None,'warning':'Partial rows are not full benchmark accuracy. Native abstract memories have no raw-note Hit@6.'}
  # Paired intersection only. InMind primary excludes the 40 recipe-selection tasks.
  for ds in expected:
   chosen={(cid,qid):v for cid,qid,v in groups.get((ds,'both'),[]) if 'evaluation' in v and (ds!='inmind' or (v.get('heldout') and qid=='query'))}
   for (dataset,m),rows in groups.items():
    if dataset!=ds or m=='both':continue
    base={(cid,qid):v for cid,qid,v in rows if 'evaluation' in v};keys=chosen.keys()&base.keys()
    if keys:report['comparisons'][ds+' both vs '+m]={'paired_questions':len(keys),'both_correct':sum(chosen[k]['evaluation']['score'] for k in keys),'baseline_correct':sum(base[k]['evaluation']['score'] for k in keys),'warning':'intersection is provisional and may be selected by runtime; wait for complete coverage'}
 dump(HERE/'night-status.json',report);return report
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 while True:
  r=status();print(json.dumps(r,ensure_ascii=False),flush=True)
  if not a.watch:break
  time.sleep(60)
