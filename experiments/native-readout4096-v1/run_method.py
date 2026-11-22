"""Unified4096-token native readout; prove retrieval matches the earlier clipped input."""
import os,sys,json,time,pickle,traceback,hashlib
from pathlib import Path
from datetime import datetime,timedelta
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];V3=ROOT/'experiments/native-source-aligned-v3';OLD=ROOT/'experiments/strict241-queryfrozen-v2';STRICT=ROOT/'experiments/strict241-v1'
os.environ['NATIVE_DEADLINE']=(V3/'deadline.txt').read_text().strip();NAME=sys.argv[1];os.environ['NATIVE_JOB']='readout4096-'+NAME;os.environ['RUN_JOB']='native-v3-readout4096-'+NAME
sys.path.insert(0,str(V3));import worker as W
import native as N
from transport import dump,budget
D=json.loads((V3/'datasets.json').read_text())['inmind'];BG=D['lines'][:240];path=OUT/'answers'/(NAME+'.json');rows=json.loads(path.read_text()) if path.exists() else {}
def main():
 shared=None;blob=None
 if NAME=='amem':
  shared=N.ingest('amem',STRICT/'prefix/amem',BG);blob=(STRICT/'prefix/amem/state.pkl').read_bytes()
 original=json.loads((OLD/'answers'/(NAME+'.json')).read_text());start_calls=N.T.calls
 for i,t in enumerate(D['tasks']):
  budget()
  if str(i) in rows:continue
  source=original[str(i)];lines=BG+[t['fact']]
  if NAME=='mem0':
   m=N.mem0(STRICT/'states/mem0'/str(i));before=W.B.unchanged_digest(m,'mem0');units,meta=N.retrieve('mem0',m,t['q'],lines);assert before==W.B.unchanged_digest(m,'mem0');N.close_memory(m,'mem0');m.db.connection.close()
  elif NAME=='amem':
   s=pickle.loads(blob);m=shared;m.memories=s['memories'];m.retriever.corpus=s['corpus'];m.retriever.embeddings=s['embeddings'];m.retriever.document_ids=s['document_ids'];m.evo_cnt=s['evo_cnt']
   m.add_note(t['fact'],time=(datetime(2026,1,1)+timedelta(minutes=240)).strftime('%Y%m%d%H%M'));units,meta=N.retrieve('amem',m,t['q'],lines)
   assert N.T.calls==start_calls,'A-MEM replay required a new construction/query-keyword model response; readout-only equivalence not established'
  else:
   units=[lines[j] for j in source['metadata']['source_ids']];meta=source['metadata']
  clipped='\n'.join('- '+str(x)[:400] for x in units[:6]) or '(none)';assert clipped==source['notes'],f'{NAME} original clipped input mismatch at{i}; do not accept a changed retrieval as a readout-only experiment'
  kept=[]
  for unit in units:
   unit=str(unit);candidate='\n'.join('- '+x for x in kept+[unit])
   if W.B.token_count(candidate)>4096:break
   kept.append(unit)
  notes='\n'.join('- '+x for x in kept) or '(none)';answer=W.lm.call(W.answer.ANS.format(notes=notes,q=t['q']))
  rows[str(i)]={'id':t['id'],'heldout':i<61,'answer':answer,'notes':notes,'applied':None,'metadata':meta,'raw_hit6':source['raw_hit6'],'context_tokens':W.B.token_count(notes),'max_context_tokens':4096,'native_units_available':len(units),'whole_units_kept':len(kept),'old_clipped_input_reconstructed_exactly':True,'reader_input_unchanged':notes==source['notes'],'native_construction_or_query_generation_new_calls':N.T.calls-start_calls,'original_answer_sha256':hashlib.sha256(source['answer'].encode()).hexdigest()}
  dump(path,rows);dump(OUT/'progress'/(NAME+'.json'),{'answered':len(rows),'total':125,'updated':time.strftime('%H:%M:%S'),'clipped_input_matches':True})
 dump(OUT/'complete'/(NAME+'.json'),{'n':125,'all_old_clipped_contexts_match':True,'no_new_native_construction_calls':N.T.calls==start_calls})
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(OUT/'errors'/(NAME+'.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
