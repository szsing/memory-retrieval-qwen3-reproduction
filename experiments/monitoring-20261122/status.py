"""One view of source reproduction, isolated pools, controls and official-prompt rescoring."""
import json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
EXPERIMENTS={'native_readout4096':ROOT/'experiments/native-readout4096-v1','shared365':ROOT/'experiments/native-source-aligned-v3','isolated241':ROOT/'experiments/strict241-v1','isolated241_frozen_queries':ROOT/'experiments/strict241-queryfrozen-v2'}
def load(p,default=None):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default
def report():
 obj={'updated':time.strftime('%Y-%m-%d %H:%M:%S'),'deadline_beijing':'2026-11-22 19:55:21','protocols':{},'warnings':['125 contains development-domain tasks;61 unseen-domain tasks must be separate.','Raw line Hit@6, simplified application, explanation-conditioned answer score and grounded application are different metrics.','241-line pools are an author-aligned isolated-candidate diagnostic, not the official full conversation timeline.','All judges are local Qwen; independent Codex review is a diagnostic, not human annotation.']}
 for name,p in EXPERIMENTS.items():
  result={f.stem:load(f) for f in (p/'results').glob('*.json')};answers=p/('answers/inmind' if name=='shared365' else 'answers')
  coverage={}
  for f in answers.glob('*.json'):
   x=load(f,{})
   if x:coverage[f.stem]={'answered':len(x),'simplified_scored':sum(r.get('applied') is not None for r in x.values())}
  errors={f.stem:load(f) for f in (p/'errors').glob('*.json')}
  obj['protocols'][name]={'results':result,'coverage':coverage,'errors':errors,'controller':load(p/'status.json',{})}
 evaluator=load(ROOT/'experiments/eval-official-v1/summary.json',{})
 obj['official_prompt_rescore']=evaluator
 obj['official_rescore_error']=load(ROOT/'experiments/eval-official-v1/error.json')
 obj['prefix_equivalence']={f.stem:load(f) for f in (ROOT/'experiments/strict241-v1/prefix-audit').glob('*fresh-equivalence.json')}
 feature=load(EXPERIMENTS['isolated241_frozen_queries']/'query-features.json',{})
 obj['query_feature_control']={k:feature.get(k) for k in ['batch_sizes','new_api_calls','validated_cache_hits']}
 tmp=OUT/'status.tmp';tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(OUT/'status.json')
 text=['# 实验监测记录','',obj['updated'],'','当前主要核验：逐题241池、固定原查询特征、保留原生结构的统一4096-token读取预算、官方提示本地复评分。原共享365池的原脚本分数保留。125题含开发域，61题单列。','', '## 官方答案应用复评分（仅列完整覆盖）','', '|协议|方法|61题|125题|','|---|---|---:|---:|']
 for protocol in EXPERIMENTS:
  for m in ['full','write_only_attr','fact_presence','rootmem','mem0','amem','hyde','rerank','plain','none','oracle']:
   v=evaluator.get('results',{}).get('/'.join([protocol,'answer',m]),{})
   if v.get('heldout61_complete'):text.append(f"|{protocol}|{m}|{v.get('correct61','—')}|{v.get('correct125','—')}|")
 text+=['','## 当前生成覆盖','', '|协议|方法|回答数|旧口径评分数|','|---|---|---:|---:|']
 for protocol,v in obj['protocols'].items():
  for m,r in v['coverage'].items():text.append(f"|{protocol}|{m}|{r['answered']}|{r['simplified_scored']}|")
 text+=['','注意：官方提示仍由本地Qwen判分；oracle使用金标准事实，非真实检索系统。grounded还要求上下文包含相应事实，结果另存status.json。旧简化判分存在偏宽，不能据此直接证明方法有效。','', '该记录不覆盖或重写任何已完成成绩。']
 (OUT/'REPORT.zh.md').write_text('\n'.join(text)+'\n');return obj
if __name__=='__main__':report()

