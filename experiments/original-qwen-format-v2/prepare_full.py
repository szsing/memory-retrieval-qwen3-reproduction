"""Expand record coverage using the author's unchanged ImplexConv sampling/parser."""
import json,re,random,hashlib,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DATA=HERE/'data';DATA.mkdir(exist_ok=True)
def write(n,x):(DATA/n).write_text(json.dumps(x,ensure_ascii=False))
def main():
 import pyarrow.parquet as pq
 src=ROOT/'experiments/full-run-v2/data';rows=[r for f in sorted(src.glob('implex-*.parquet')) for r in pq.read_table(f).to_pylist()];assert len(rows)==1550
 users=[]
 for idx,r in enumerate(rows):
  c={k:v for k,v in r['conversation'].items() if v};qa=r['qa'];ev=sorted({str(i) for q in qa for i in q['retrieved_conv_ids']});rng=random.Random(idx);others=[k for k in sorted(c) if k not in ev];keep=ev+rng.sample(others,min(30,len(others)));sess={k:re.findall(r'(?m)^Speaker1: (.*)',c[k]) for k in keep}
  users.append(dict(row=idx,sessions=sess,qa=[dict(q=q['question'],ev=[str(i) for i in q['retrieved_conv_ids']],trait=q['opposed_implicit_reasoning']) for q in qa]))
 original=json.loads((ROOT/'rsi-memory/bench_data/implexconv/implexconv_opposed_trim.json').read_text())
 assert users[:30]==original['val'],'Original val data differs from frozen full source'
 assert users[800:830]==original['test'],'Original test data differs from frozen full source'
 write('implexconv_full_same_protocol.json',{'val':users,'test':users})
 missing=sum(not any(u['sessions'].get(sid) for sid in q['ev']) for u in users for q in u['qa'])
 d=json.loads((ROOT/'rsi-memory/bench_data/lucid/lucid.json').read_text());eligible=[r for r in d if any(t.get('has_answer') for s in r['latent_sessions'] for t in s if t['role']=='user')]
 write('full-data-audit.json',{'implex_records':1550,'implex_questions':sum(len(u['qa']) for u in users),'implex_original_subsets_equal':True,'implex_no_reachable_target_with_original_parser':missing,'implex_policy':'exact author evidence sessions +30 other sessions; original Speaker1 regex and 1000-char scoring truncation retained, not full-history evaluation','lucid_raw_records':len(d),'lucid_eligible_records':len(eligible),'lucid_excluded_no_user_answer_ids':[r['query_id'] for r in d if r not in eligible],'lucid_policy':'full eligible query coverage in query-id-group folds <=240 rows (oversized groups kept together), same latent+100-other-session pool; exclude own answer session IDs from other-session donors in full extension only','inmind_full_tasks':125,'inmind_full_is_descriptive':True})
 print((DATA/'full-data-audit.json').read_text(),flush=True)
if __name__=='__main__':main()
