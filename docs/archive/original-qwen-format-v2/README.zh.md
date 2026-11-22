# 原作者协议：Qwen3-8B 输出格式修复复验

原作者快照为 d62aa408a4e0b9515eed9313663a025a07b08497。上一轮结果保留在 original-qwen-night-v1。本轮使用独立源码快照与空缓存，默认禁止复用上一轮 LLM 输出。

改动仅在模型接口与解析适配层：对原本要求 JSON 的调用添加 JSON Schema 约束；验证完整编号、字段、类型；支持完整 NDJSON、代码围栏与十进制字符串编号；禁止把缺失项补成空属性。连续两次失败则明确标记该任务失败。合法的 p=[] 独立计数，不当作解析失败。原 answer.py 提示使用 Item 1..N，但消费端按 0..N-1 读取，适配层显式转换并记录。

Schema 约束会影响生成分布，本轮属于格式适配后的本地模型复验。原提示文本、批量大小、temperature=0、max_tokens=2048、模型、数据、候选池、权重、属性版本和排序算法均保持不变。原作者源码 32 个文件逐一校验哈希。不会为了避免截断擅自加 token 或拆批。格式失败、截断、thinking、缺失编号均记录。

相同子集：InMind 61 个跨域任务，LUCid test 240 题，ImplexConv test 81 题，另保留原验证集、harm 与历史消融。先核对纯 dense 与原作者数字一致，再跑 HyDE、完整方法和 nojoin 对照。源码 nojoin 的既有实现保持原样，其结果不自动解释为严格的双侧贡献消融。

自动扩大量的筛选门槛：三套核心子集上，完整方法均不低于 dense、HyDE、nojoin；至少一套相对 HyDE 提升 5 个百分点且净增至少 3 个命中；没有格式失败、thinking 或截断，缺失项和 None 比例不超过 1%。这是计算预算分配门槛，不是显著性检验或有效性的最终证明。

门槛通过即按原作者候选池协议扩展记录覆盖，8 小时预算内保存每一折结果。扩大量不等于官方完整长历史评测：ImplexConv 保留证据会话加 30 个干扰会话；LUCid 保留 latent 加 100 个其他会话；InMind 扩至 125 个任务，包含原开发任务，因此只能描述覆盖，不能称为独立测试成绩。仅共同完成的折可作公平比较。全量 InMind 调度将逗号分隔的 bed 标记正确识别为 inmind，防止名为 full 实际仍只有 61 题。

入口 run_night.sh；状态 status.json；门槛 alignment-gate.json；汇总 REPORT.zh.md / summary.json；原始请求响应 api-responses.jsonl；格式事件 events.jsonl；分任务结果 results/。
