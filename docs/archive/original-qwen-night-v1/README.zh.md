# 原作者脚本切换本地 Qwen3-8B：八小时实验

源快照：d62aa408a4e0b9515eed9313663a025a07b08497（2026-11-17 Round5）。线上链接当前404，使用此前已有快照。32个原始源码/文档/脚本以SHA256核对，副本放在服务器本目录snapshot；上游源码没有改写。截图里后续RootMem结果不在该快照，本轮不声称复现那部分。

服务器目录：`/root/autodl-tmp/memory-retrieval-indirect-requests/experiments/original-qwen-night-v1`。

## 唯一模型替换及必要运行边界

Haiku backend替换为仓库已有OpenAI兼容接口，服务本地Qwen3-8B BF16（revision b968826d9c46dd6066d109eabc6255188de91218）。vLLM0.11.0；服务端模板关闭thinking。保持原OpenAI分支temperature=0、max_tokens=2048，不额外加schema，不修改提示，不把原批处理改成单条，不改变RRF、并列处理、BGE query prefix、截断长度、门控、pin或picker回退。

Haiku原CLI分支并未显式指定输出token上限，因此无法声称2048与Haiku历史实际上限完全相同；2048是原仓库提供的OpenAI分支设置。这次不因模型输出异常而自行提高到8192。

外层instrument.py仅记录调用、完整原始响应、finish_reason、thinking字段、JSON文本解析状态、原jcall返回None、by_id缺失/空列表和methods.batched的None。原解析失败默认处理仍执行，但日志会明确展示，不能把技术失败误作有效空属性。预算中断除外，包装器返回值和HTTP请求体已通过与原函数对照检查。

数据未随上游仓库完整发布，使用此前下载并冻结的同名数据，哈希在data-source-lock.json。先以模型无关Dense结果作为输入/检索实现对齐检查：InMind official1/61、dense1/61、direct73/85，LUCid42/240、Implex15/81；不一致则停止并记录，不能把差异归因于Qwen。

## 原子集与原配方

- InMind跨域61题，原241行式官方评分及365行竞争池、85题direct检查原样保留；40个开发任务不重新选权重。
- LUCid hash固定val120/test240，latent sessions+100个共享其他用户会话。
- ImplexConv test原rows800–829，共81题；val原rows0–29，共79题；证据会话+30个其他会话，原Speaker1正则和1000字符裁剪保留。
- plain、HyDE、Round5完整方案与原nojoin重排对照：`hyde+prop --mix pick --w 8 --pv 2 --ctrl wide/nojoin`。
- Round2基础机制对照使用原bridge.score_set：`prop --mix rrf --w 2 --side both/line`及`--ctrl null`，不是重新调参数。
- 原answer.py的80词回答、top6每条400字符、10题一批judge保持。它的InMind答案上下文来自365行池，不能误读为241行式official检索上下文。回答检查是单独记录的本地Qwen自评。

调用原bench.run等入口函数返回逐题结果，设置与原CLI一致；结果不混用snapshot内继承的Haiku历史日志。新的结果放在本目录results，原始提示响应放api-responses.jsonl，解析观测放events.jsonl。

已知上游quirks保持而非偷偷修复：RRF全局排名后再限候选池；并列采用原argsort；nojoin只改picker候选而未彻底移除join后备排序。因此本轮是原代码模型迁移复验，不是已修正所有实现问题的最终论文评测。

## 自动扩大全量：看结果前冻结

这是决定是否值得进一步跑的工程筛选条件，不是论文显著性或创新性证明：

1. 原始Dense基线全部匹配。
2. 三个核心test子集上完整方案命中数均不低于plain、HyDE和同模型原nojoin重排；至少一个子集相对HyDE提升≥5个百分点且净增≥3题。
3. 核心子集调用无length截断、无thinking输出；原jcall最终None率、条目缺失率及批处理None率均≤1%。Round2早期v1诊断单独报告，不替代Round5核心质量检查。

满足后立即进入full队列；不满足不扩大全量，而完成原answer checks并明确记录阻断原因。不会看到测试结果后改阈值、权重或提示。

## “全量”的准确含义

扩展的是全记录/全合格题目覆盖，保留作者的数据取样与打分定义，**不是切换到官方完整历史评测协议**。

- InMind125任务，原365行池，包含原开发任务，作为描述性全量结果；跨域61仍单独保留。
- ImplexConv全1550记录/3633题，每30记录一组复用原候选构造，保持原30-other-session采样与用户行正则。原val/test两个30记录子集须与从冻结parquet重建的值逐项相同，才允许扩展。
- LUCid原1936条中，原函数要求有user-role has_answer，合格1903条；33条排除理由/ID记录。按query_id分组折叠，每折约240条，保持latent+100其他会话；扩展时为避免全量样本自己的证据出现在其他用户干扰池，排除当折query_id及其证据session ID。这项不可避免的数据扩展适配单独注明，不改排序算法。

全量按折交错运行plain/HyDE/full/rerank，两套大数据共同推进。所有对照同折对比，不能将不同完成子集分数直接相减。

## 时间、检查点与查看

默认8小时，deadline-unix.txt记录实际截止。到点停止新模型请求，正在执行的请求可结束；中断任务保留原模型缓存，后续可续跑。至少留3GiB磁盘。不会关机或删除旧实验。

主状态status.json；基线对齐dense-baseline-alignment.json；扩展决策alignment-gate.json；全量数据审计data/full-data-audit.json；逐任务logs。supervisor.lock防止重复启动。

```
cd /root/autodl-tmp/memory-retrieval-indirect-requests
nohup env MEM_PROJECT_ROOT="$PWD" bash experiments/original-qwen-night-v1/run_night.sh > experiments/original-qwen-night-v1/supervisor.log 2>&1 &
```

