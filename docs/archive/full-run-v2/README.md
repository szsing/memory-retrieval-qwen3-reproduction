# 本地开放模型全量实验队列 v2

本目录对应服务器 `/root/autodl-tmp/memory-retrieval-indirect-requests/experiments/full-run-v2`。
主机 RTX 4090；Qwen3-8B BF16，revision `b968826d9c46dd6066d109eabc6255188de91218`；vLLM 0.11.0；关闭 thinking；temperature 0、seed 42；BGE-small-en-v1.5 CPU。没有调用付费 API。

## 数据和主指标

- InMind 原始 125 任务，每任务同一完整冻结状态下测试间接/直接两题，共250。40个开发任务选配方，85个不重叠任务为主测试集。原始官方背景的47个会话与目标插入位置保留。连续用户/助手交互切块，不跨会话；这是统一切块的本地对照实验，不能称作官方487单轮Naive RAG完全一致复现。
- LUCid S：冻结官方 revision `2431979d0f9e8ec9688a33cf21a77f6c967e5a9c`，1936条评测记录，每行50个会话位置。仅1101种query_id，因此使用行号+query_id作为记录标识，重复记录不删；统计推断应按重复query或同源用户聚类。4条记录含重复session ID，因此唯一会话ID数49，并未删除第50个位置。参考答案、has_answer 等只留给评测。
- ImplexConv-opposed：两个完整parquet，1550条记录，3633个QA；117条无QA记录在数据审计中保留，不进入题数分母。完整会话解析Speaker1/Speaker 1及Assistant；无标签说明文字保留为助手文本，不用于用户侧属性生成；空turn保留。未截取前30条、前30会话或前1000字符。
- 全部输入约149万次交互块出现、约80.8万种不同块，完整记忆系统建库无法保证一夜全部完成。先在三套数据各校准一个完整case，然后优先完成InMind，再推进S、Implex。所有方法输入完整历史；不以裁剪历史换取“全量”标签。
- LUCid B仅在S全部15个方法的检索和回答评分完成后、仍有时间及至少10GiB空间时启动。B同一官方revision，1936记录×200会话；SHA256 `ed1d088c395064e58d8c55438174fe696a4bdd56c323df37468dbc98d40f5836`，从官方HF tree API核对。未满足条件时记录waiting_for_full_S。

## 方法与冻结选择

开发消融在测试前锁定，选择 `both`，RRF常数60、属性权重2、top6、4096 token上下文。不得根据LUCid/Implex结果重新选择超参。

实际计算15个方法输出：Dense、HyDE、Dense top12 LLM重排；both、write_only、query_only、shuffled、gated_pin、picker、fact_prior、gated_fact_prior；Mem0、A-MEM、RootMem、RootMem+both。消融共享特征以节省计算，运行时不能直接用于独立方法成本比较。

原始属性/HyDE/picker提示保留；每次HTTP只处理一个note或一个query。JSON schema通过API解码参数传递，不插入额外语义提示。night transport包括异常路径均不改写原messages；拒收早期尝试中发生提示变更的缓存，截断和格式失败单独记录。写侧只用用户内容，原始检索保留全部角色。每个case内独立排名、并列分数取平均rank；原始记录不因共享缓存进入其他case候选池。

Mem0使用官方PyPI `mem0ai==0.1.98`，.venv-baselines，Qdrant逐case隔离、完整原生add/search；A-MEM使用官方paper repo source commit `f9a173dec325d84fcc97e7828775c7b241c33466`，保留原生分析、关联、演化、consolidation、查询关键词预处理和链接扩展。统一Qwen/BGE是明确的后端适配，不是原论文模型分数复现。

RootMem使用官方IMLogic独立build/router/context实现，源文件hash见vendor-night-source-lock.json。原始用户文本适配为memory_points（逻辑次序标记，不伪造事件日期），原生过滤与排序仍执行。批次最多60条，超出32K上下文时缩小整个批次、不切掉记忆，输出上限8192；本地BGE替代外部MemOS检索；共同reader和4096 token整单元预算。因此属于官方算法的受控适配，不称作原论文完整配置复现。

## 评测

原始块检索报告session Hit@6和session Recall@6，InMind另外报告精确目标块Hit@6。Mem0等抽象记忆不伪造原始块Hit@6；主跨系统指标为相同reader的端到端结果。

InMind沿用官方answer/application/naive/target-recall提示；LUCid沿用官方MCQ/suitability评分分流及提示；ImplexConv AST提取官方二分类判分提示，并沿用原BLEU1/2/3和ROUGE-L公式。替换为本地Qwen reader/judge；Implex通用reader提示与原实验不同，LUCid判分加API JSON约束。统一reader最多2048输出token；LUCid本地judge最多1024以避免其原200上限在Qwen上截断。相同骨干自评有偏差，正式论证仍需独立judge或人工一致性抽查。

评分错误与未完成严格区分，不记为0。night-status.json总是保留官方分母、检索完成数、评分完成数、正确数、错误数；部分完成的accuracy仅说明已评子集，不作为全量结论。配对交集也仅作临时诊断，存在运行时选择偏差。

## 运行与续跑

```
cd /root/autodl-tmp/memory-retrieval-indirect-requests
nohup env MEM_PROJECT_ROOT="$PWD" bash experiments/full-run-v2/run_night.sh > experiments/full-run-v2/night-supervisor.log 2>&1 &
```

默认8小时窗口；deadline写入night-deadline-unix.txt。时间达到后在模型新请求前暂停，至少留3GiB磁盘。当前正在生成的请求允许完成。已完成case不重跑、LLM原始请求/响应缓存保留，RootMem逐批检查点、完整native建库状态持久化；未完成Mem0/A-MEM建库不能保证逐步原样恢复，改从已验证缓存重建，并保存中断目录。不是全量完成才停止一切。

再次执行同一脚本时给新的8小时窗口，将paused_budget/running任务重新排队。三次连续case失败会暂停对应方法剩余case并标blocked_after_failures，需要检查日志修复后显式重置；不得把它们算成成功。

审计：data-audit.json、night-preflight.json、prompt-fidelity-preflight.json、vendor-night-source-lock.json、data/download-manifest.json。进度：night-status.json；原始结果：night.sqlite3；原生状态：night-state；逐case错误：night-case-logs；完整原始模型请求响应：transport-*.jsonl及llm-cache.sqlite3。

**full-run-v1 是已撤销的适配诊断，不能与本目录正式v2结果混用。**
