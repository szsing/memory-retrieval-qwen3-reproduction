"""Assemble a source-grounded Chinese report from completed artifacts."""
from night_data import *
from collections import defaultdict
P=HERE/'controlled-distractors-v4'
def report():
 rows=json.loads((P/'retrieval-results.json').read_text());ids={r['task_id'] for r in rows if r['level']==10 and r['seed']==42};pool=json.loads((P/'construction-audit.json').read_text());tests=json.loads((P/'retrieval-primary-tests.json').read_text())
 names={'both':'完整双侧','write_only':'仅写侧','fact_prior':'属性有无','dense':'Dense'}
 lines=['# 多事实干扰控制实验','', '固定此前开发集选定的配方：RRF=60、属性权重2、top6、上下文4096 token；比较前锁定10条干扰为主设置。主比较覆盖79个未参与原配方选择的间接请求。','', '这是人工构造的持续性家居/工作/社团约束压力测试，不是原始InMind、LUCid或ImplexConv全量分数。没有使用本轮命中率来选干扰事实或调整参数。','', '## 构造与覆盖','', '- 最初的批量模型标注出现漏条及明显错误的互斥关系，在运行检索前被弃用；一般设备规格池48条仅3条得到非空属性，也未跑多干扰检索。最终使用预先写好的48条长期约束，仍采用原写侧提示，其中10条产生非空属性。','- 对直接影响当前请求的候选使用人工逐项排除；剩余记忆为非目标背景约束，可能影响外围建议，不能宣称全是经过独立专家确认的完全不相关事实。','- 每个方法共享相同原始完整历史、候选记忆、随机插入位置、预算和检索参数。目标记忆与247个原始交互块保留。','- 5条干扰覆盖83/85题，10条覆盖79/85题；20条因合格事实池不足未运行。覆盖不足的请求不计为成功，也没有按性能筛选删除。','- 随机种子42/43/44；10条设置使用全部10条合格事实，三个种子只改变插入位置，实际排名结果相同，不能当成三份独立样本。','- 79题中37题的目标事实能提取属性，其余42题目标属性为空。分层仅用于分析，不是按已知目标属性决定是否启用方法。','', '## 同一79题的检索结果（预设seed42）','', '|方法|0条干扰 Hit@6|5条干扰 Hit@6|10条干扰 Hit@6|10条干扰 Hit@1|','|---|---:|---:|---:|---:|']
 for m in names:
  hit=[]
  for k in [0,5,10]:
   rr=[r for r in rows if r['task_id'] in ids and r['level']==k and r['seed']==42 and r['method']==m];assert len(rr)==79;hit.append(sum(r['hit6'] for r in rr))
  r10=[r for r in rows if r['task_id'] in ids and r['level']==10 and r['seed']==42 and r['method']==m]
  lines.append('|'+names[m]+'|'+'|'.join(str(n)+'/79' for n in hit)+'|'+str(sum(r['hit1'] for r in r10))+'/79|')
 lines+=['', '零干扰设置在全部85题逐字复现既有上下文，Both/Write-only/Fact-prior各47/85、Dense7/85。上表仅取共同79题，避免把不同样本的分母变化误认为干扰导致的性能变化。','', '## 主比较的证据强度','']
 for m in ['write_only','fact_prior']:
  v=tests['both vs '+m];lines.append('- 完整双侧 vs '+names[m]+'：命中'+str(v['ours_hit6'])+' vs '+str(v['control_hit6'])+'，配对胜/负='+str(v['wins'])+'/'+str(v['losses'])+'；双侧精确McNemar p='+str(v['mcnemar_case_level_exact_unadjusted'])+'，按目标事实聚类的符号翻转检验p='+str(v['target_fact_block_signflip_p_unadjusted'])+'。')
 lines+=['', '与仅写侧的差异由5个案例支持，尚不能稳定推广；与更简单的属性存在性对照相比仅净多1题。当前检索结果不支持宣称双侧匹配有稳定、普遍的独立贡献，也不能据此断言它在所有场景无用。检验均为探索性、未做多重比较校正。','', '## 提取遗漏的分层结果','', '在共同79题中，37题目标属性非空：0干扰时三种属性方法均37/37命中，10干扰时完整30/37、仅写侧25/37、存在性29/37。其余42题目标属性为空：0干扰时三者均6/42，10干扰时三者全部0/42，而Dense仍为6/42。该配置会把有属性的干扰事实推到没有属性的目标事实之前，抽取覆盖率是明显瓶颈。','', '## 案例与局限','', '- 玉米烹饪请求与“两岁孩子”：完整双侧把目标排第3，仅写侧第9，属性有无第11，是查询侧隐含属性可能提供帮助的例子。','- 电工资格考试：完整双侧第9，属性有无第6，显示查询扩展也可能损害目标召回。','- 玉米案例的回答检查：虽然完整方法将“两岁孩子”排进检索结果且实际送入上下文，reader仍给出一般烹饪建议并忽略该条件。application判为0、target-recall判为1；应用判分理由中“上下文没有该事实”的表述与实际输入不符，说明同模型判分理由也需要独立核验。','- 干扰池只有10种非空约束，且大部分为显式禁止性规则；这不能替代多用户真实长期对话上的检验。','- Bootstrap对少量不一致案例可能过于乐观，因此同时给精确配对检验和按事实分组的检验，不用重复种子虚增样本量。','', '## 主设置回答应用评测','']
 if (P/'qa-summary.json').exists():
  q=json.loads((P/'qa-summary.json').read_text());pairs=json.loads((P/'qa-paired.json').read_text());lines+=['统一原InMind回答/应用/目标召回提示，reader与judge均为本地Qwen3-8B；所有方法reader输出上限4096，judge2048。相同骨干自评并非独立人工验证。','', '|方法|已评/应评|应用成功|上下文目标召回判分|','|---|---:|---:|---:|']
  for m in ['both','write_only','fact_prior']:
   v=q['all85'][m];lines.append('|'+names[m]+'|'+str(v['n'])+'/79|'+str(v['application_correct'])+'/'+str(v['n'])+'|'+str(v['target_recall_correct'])+'/'+str(v['n'])+'|')
  for k,v in pairs.items():
   if k.startswith('all85'):lines.append('\n'+k+'：应用差值='+str(v['delta_application'])+'，聚类bootstrap95%CI='+str(v['cluster_bootstrap_ci95'])+'，未校正精确p='+str(v['mcnemar_exact_unadjusted'])+'。')
 else:lines.append('回答判分尚在运行；当前不能把检索改善解释为回答质量已被证明提高。')
 lines+=['', '主要审计：protocol.lock.json、qa-protocol.lock.json、construction-audit.json、quality-check.json、retrieval-primary-tests.json、retrieval-paired.json、qa-run-finished.json。原始检索结果和每题回答/判分均保留。']
 (P/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':report()
