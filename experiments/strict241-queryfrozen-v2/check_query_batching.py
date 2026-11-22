import os,sys,json,statistics
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1];os.environ['RUN_JOB']='diagnostic-query-batching';sys.path.insert(0,str(root/'experiments/native-source-aligned-v3'))
import worker as W
import methods as M
W.bridge.PV[0]=2;f=json.loads((p/'query-features.json').read_text());single_h=[];single_p=[];single_g=[];before=dict(W.lm.STATS)
for q in f['queries']:
 single_h+=M.gen_hyde([q]);single_p+=W.bridge.props_reqs([q]);single_g+=W.bridge.gate_reqs([q])
result={'diagnostic_only':True,'original_global_batch_sizes':f['batch_sizes'],'new_api_calls':W.lm.STATS['calls']-before['calls'],'cache_hits':W.lm.STATS['cached']-before['cached'],'n':125,'global_gate_help':sum(f['gate']),'single_gate_help':sum(single_g),'gate_disagreements':sum(a!=b for a,b in zip(f['gate'],single_g)),'global_query_properties_nonempty':sum(bool(x) for x in f['properties']),'single_query_properties_nonempty':sum(bool(x) for x in single_p),'property_lists_identical':sum(a==b for a,b in zip(f['properties'],single_p)),'hyde_text_identical':sum(a==b for a,b in zip(f['hyde'],single_h)),'global_hyde_median_chars':statistics.median(map(len,f['hyde'])),'single_hyde_median_chars':statistics.median(map(len,single_h))}
(p/'query-batching-diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
