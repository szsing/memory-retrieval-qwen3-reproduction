"""Readout-only sensitivity: reuse RootMem state/routes; prove old clipped notes exactly match."""
import os,sys,json,time,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];V3=ROOT/'experiments/native-source-aligned-v3';BASE=ROOT/'experiments/strict241-v1'
os.environ['NATIVE_DEADLINE']=(V3/'deadline.txt').read_text().strip();os.environ['RUN_JOB']='native-v3-root-native4096';os.environ['NATIVE_JOB']='root-native4096'
sys.path.insert(0,str(V3));import worker as W
from transport import dump,budget
D=json.loads((V3/'datasets.json').read_text())['inmind'];BG=D['lines'][:240]
W.ret.CDIR=str(ROOT/'experiments/strict241-queryfrozen-v2/embedding-cache');W.ret.save=lambda:None
path=OUT/'answers/rootmem.json';rows=json.loads(path.read_text()) if path.exists() else {};B=W.B
for i,t in enumerate(D['tasks']):
 budget()
 if str(i) in rows:continue
 while True:
  budget();source=json.loads((BASE/'answers/rootmem.json').read_text())
  if str(i) in source:break
  time.sleep(5)
 original=source[str(i)];matrix=json.loads((BASE/'states/rootmem'/str(i)/'matrix.json').read_text());domains=original['metadata']['activated_domains'];active=[s for s in matrix['agent_skills'] if s['Skill_Domain'] in domains]
 lines=BG+[t['fact']];scores=W.ret.embed(lines)@W.ret.embed([t['q']],query=True)[0];ids=np.argsort(-scores)[:6].tolist();raw=[lines[j] for j in ids];raw=raw[:min(6,max(0,30-len(active)))]
 units=[json.dumps(s,ensure_ascii=False) for s in active]+raw
 clipped='\n'.join('- '+s[:400] for s in units[:6]) or '(none)';assert clipped==original['notes'],f'Cannot reconstruct original reader input for{i}; no readout comparison accepted'
 kept=[]
 for unit in units:
  candidate='\n'.join('- '+s for s in kept+[unit])
  if B.token_count(candidate)>4096:break
  kept.append(unit)
 notes='\n'.join('- '+s for s in kept) or '(none)'
 response=W.lm.call(W.answer.ANS.format(notes=notes,q=t['q']))
 rows[str(i)]={'id':t['id'],'heldout':i<61,'answer':response,'notes':notes,'applied':None,'original_notes_sha256':hashlib.sha256(original['notes'].encode()).hexdigest(),'native_matrix_sha256':hashlib.sha256(json.dumps(matrix,sort_keys=True).encode()).hexdigest(),'original_input_reconstructed_exactly':True,'context_tokens':B.token_count(notes),'max_context_tokens':4096,'active_skills':len(active),'complete_skill_units_retained':min(len(active),len(kept)),'original_logic_region_keys_visible':original['notes'].count('Logic_Aware_Decision_Region'),'new_logic_region_keys_visible':notes.count('Logic_Aware_Decision_Region'),'readout_sensitivity_only':True}
 dump(path,rows);dump(OUT/'progress.json',{'answered':len(rows),'total':125,'latest_i':i,'updated':time.strftime('%H:%M:%S'),'all_original_clipped_contexts_match':True})
 if i==7:dump(OUT/'preflight.json',{'n':8,'original_clipped_notes_reconstructed_exactly':True,'same_frozen_rootmem_matrices_and_routes':True,'no_memory_build_or_router_calls':True,'reader_model_unchanged':True,'changed_readout_budget':'whole units up to4096 tokens, versus6 units*400 chars'})
dump(OUT/'complete.json',{'n':125,'all_reader_inputs_verified':True,'sensitivity_not_primary_same_budget_comparison':True})
