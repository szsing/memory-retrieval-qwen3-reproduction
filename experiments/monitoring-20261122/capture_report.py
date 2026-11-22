"""Capture actual coverage and scores; never turn missing results into zero."""
import json,time
from pathlib import Path
O=Path(__file__).resolve().parent;E=O.parent

def load(p,default=None):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

S=load(E/'eval-official-v1/summary.json',{});P='native_readout4096';methods=['full','write_only_attr','fact_presence','mem0','amem','rootmem','hyde','rerank','plain','none','oracle']
primary={m:{metric:S.get('results',{}).get(P+'/'+metric+'/'+m,{}) for metric in ['answer','grounded']} for m in methods}
coverage={};results={}
for b,n in [('implex',81),('lucid',240)]:
 coverage[b]={};results[b]={}
 for m in ['full','mem0','amem','rootmem','hyde','rerank','plain']:
  c=load(E/'native-source-aligned-v3/contexts'/b/(m+'.json'),{});a=load(E/'other-native-readout4096-v1/answers'/b/(m+'.json'),{})
  coverage[b][m]={'retrieved':len(c),'answered':len(a),'total':n}
  r=load(E/'other-native-readout4096-v1/results'/f'{b}-{m}-{n}.json')
  if r:results[b][m]=r
obj={'captured':time.strftime('%F %T'),'deadline_beijing':'2026-11-22 19:55:21','primary':primary,'supplemental_coverage':coverage,'supplemental_diagnostic_results':results,'priority':load(O/'priority-status.json'),'controller':load(O/'continuation-status.json'),'primary_score_provenance':load(O/'primary-score-provenance-audit.json'),'root_truncation_recovery':load(O/'root-lucid-truncation-recovery.json'),'pool_audit':load(O/'other-pool-audit.json'),'official_full_timeline_or_LUCid_S_claimed_complete':False}
failures=[]
for line in (E/'native-source-aligned-v3/events.jsonl').open():
 try:r=json.loads(line)
 except:continue
 if str(r.get('job','')).startswith('continuation-') and r.get('kind')=='attempt_failed':failures.append(r)
obj['continuation_native_attempt_failures']=failures
obj['common_prefix10']=load(O/'common-prefix10-results.json')
obj['common_prefix20']=load(O/'common-prefix20-results.json')
obj['final_exit_audit']=load(O/'final-exit-audit.json')
obj['root_matrix_audit']=load(O/'root-completed-matrix-audit.json')
obj['common_prefix20']=load(O/'common-prefix20-results.json')
obj['final_exit_audit']=load(O/'final-exit-audit.json')
obj['root_matrix_audit']=load(O/'root-completed-matrix-audit.json')
truncations=sum('Truncated/non-stop' in r.get('error','') for r in failures)
(O/'verified-progress-snapshot.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2))
t=['# 实验进度与结果记录','',obj['captured'],'','本轮InMind125题及两项本地模型复评分均已完成；61道未见域题单列。当前结果不支持双侧匹配具有稳定独立贡献。','', '|方法|答案应用61|答案应用125|grounded61|grounded125|','|---|---:|---:|---:|---:|']
labels={'full':'完整方法','write_only_attr':'仅写侧','fact_presence':'属性有无','mem0':'Mem0','amem':'A-Mem','rootmem':'RootMem'}
for m in labels:
 a=primary[m]['answer'];g=primary[m]['grounded'];t.append(f"|{labels[m]}|{a.get('correct61','未完')}|{a.get('correct125','未完')}|{g.get('correct61','未完')}|{g.get('correct125','未完')}|")
t+=['','指标：答案应用检查回答是否给出金标准解释要求的个性化后果；grounded另要求上下文包含对应事实。两套判分提示在少量题上会判断不一致，分歧待独立复核。125题含开发域，不能替代61题泛化结果。逐题241池、原查询特征冻结、完整原生单元4096-token预算、Qwen3-8B BF16/无thinking/temperature0、共同回答提示。评分用官方InMind提示文本，但判分后端仍为本地Qwen，尚需独立复核。2750条复评分的答案与上下文哈希已一致核验。此候选池实验不是官方完整会话时间线成绩。','', '## 补充子集（原简化判分，仅作诊断）','', '|子集|方法|检索流程完成|回答完成|简化判分完整结果|原始Hit@6|','|---|---|---:|---:|---:|---:|']
for b,n in [('implex',81),('lucid',240)]:
 for m in ['full','mem0','amem','rootmem','hyde','rerank','plain']:
  c=coverage[b][m];r=results[b].get(m,{});score=r.get('diagnostic_source_judge_correct','未完');hit=r.get('raw_hit6');hit='—' if hit is None else hit;t.append(f"|{b}|{labels.get(m,m)}|{c['retrieved']}/{n}|{c['answered']}/{n}|{score}|{hit if hit is not None else '不适用'}|")
t+=['','未完成项没有记为0；原生检索输出不能直接与原始行Hit@6混为一谈。ImplexConv为原脚本81题/30个池；LUCid为原脚本240题/240个池，每池自身相关会话加100个无关会话，不是官方S全量。','', f'补充运行共记录{truncations}次非正常结束/截断尝试，内容均不被直接采用。其中首个LUCid RootMem的16384-token重复长输出，已用完全相同请求重试成功，5382 tokens、34个不同技能，原配置未变。原始失败与恢复记录均保留；其他尝试详情见JSON快照。','', '原生系统使用官方源码组件及本地模型/BGE适配；不宣称与各论文配置、朋友未提供的后续RootMem组合脚本完全一致。各池按原顺序构建，公共后缀不重排成前缀，不跨不同记忆状态偷用构建结果。','', '主结果、旧版本、完整调用日志、候选池哈希与检查点均保留。']
if obj['common_prefix10']:
 c=obj['common_prefix10']['rows'];t+=['','## 同题前10题复现诊断','', '固定顺序的首个完整10题判分批次；不是随机样本，不代表全量。']
 t+=['、'.join(m+' '+str(c[m]['source_diagnostic_correct'])+'/10' for m in ['full','mem0','amem','rootmem','hyde','rerank','plain'])+'。仍是原简化判分。']
if obj['common_prefix20']:
 c=obj['common_prefix20']['rows'];t+=['','## 扩展的相同前20题诊断','', '相同规则续完第二个10题批块，前10题保留；仍不是随机样本或全81题成绩。']
 t+=['、'.join(m+' '+str(c[m]['source_diagnostic_correct'])+'/20' for m in ['full','mem0','amem','rootmem','hyde','rerank','plain'])+'。仍是原简化判分。']
if obj['common_prefix20']:
 c=obj['common_prefix20']['rows'];t+=['','## 同题前20题扩展诊断','', '在两套原生系统共同覆盖20题后追加第二个完整10题判分批次；不依据分数选择是否扩展，前10题记录保留。']
 t+=['、'.join(m+' '+str(c[m]['source_diagnostic_correct'])+'/20' for m in ['full','mem0','amem','rootmem','hyde','rerank','plain'])+'。仍为固定顺序小样本与原简化判分。']
if obj['final_exit_audit']:
 a=obj['final_exit_audit'];timeouts=sum('Timeout' in r.get('error','') for r in failures);t+=['','## 本轮收尾','',f'新调用已在19:55运行时限停止，已发请求完成或超时后收尾。记录到{timeouts}次超时：LUCid RootMem末次长调用600秒超时，截止后没有再次发请求，未将该未完成池评分。', 'A-Mem有1次无效链接提议被原生校验拒绝，无无效边插入；两个Mem0进程预算后未退出，在确认无在途请求后清理。BudgetEnd与手动预算清理不计为模型准确率错误，缺失结果不计零。', '全部实验工作进程已退出；推理服务与已保存数据、有效模型响应缓存、完成的原生状态及RootMem分批检查点保留。']
if obj['root_matrix_audit']:
 a=obj['root_matrix_audit'];t+=['',f"最终检查{a['completed_matrices_checked']}个已完成RootMem矩阵：其中{len(a['exact_duplicate_skill_domain_matrices'])}个LUCid矩阵有同名技能重复（第2题，3个技能名称各重复一次）。JSON可解析，属于输出内容质量问题；原始结果保留，未事后去重改分。InMind及ImplexConv完成矩阵没有发现同名重复。LUCid原生复现仍需继续核查，不宣称全部通过。"]
(O/'MONITORING_REPORT.zh.md').write_text('\n'.join(t)+'\n');print(obj['captured'])

