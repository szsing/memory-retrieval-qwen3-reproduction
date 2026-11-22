# 三个原生记忆系统的同模型对照

服务器目录 experiments/native-source-aligned-v3。模型均为本地 Qwen3-8B BF16，同一冻结 revision 与 vLLM 服务；关闭 thinking、temperature=0。沿用已安装的 Mem0 0.1.98、A-MEM paper repository f9a173dec325d84fcc97e7828775c7b241c33466 与 IMLogic 8823defc18ec94e377a835695ce9180678eb2b0e 的 RootMem standalone build/router/context。具体运行源码另存哈希。

这是官方算法的受控本地适配，不是原论文分数复现。Mem0 保留 infer=True 的提取、检索旧记忆和 ADD/UPDATE/DELETE/NONE；A-MEM 保留 note analysis、关联、演化与原测试脚本的 query keyword preprocessing、linked retrieval；RootMem 保留原构建和 router 提示、增量矩阵与 DYNAMIC_MATRIX_RAG，外部 MemOS 检索替换为当前 BGE-small 原文检索。

格式约束只通过 API JSON Schema 传递，不改写原提示。Mem0 引用 ID 只能来自原提示提供的临时旧记忆编号；ADD 使用独立分支。无效输出、截断、底层吸收异常均作为复现失败，不当成零分。RootMem 路由 fallback 单独拒收。Mem0/A-MEM 输出上限2048；RootMem 构建8192，以容纳其原生累积矩阵，router2048。重试最多两次且提示不变。RootMem 按原参考60条一批，只有输入超过本地32K窗口才减少整条记忆的批次数；不截断或丢弃源记忆。

## 数据和分数边界

优先 InMind。现有官方式每题241候选池与365条共享池是两个不同实验。本轮原生系统主对照采用与现有 answer.py 一致的365条共享池：240背景条目+125事实；先评原testB 61题，通过技术检查后扩到125题。所有方法的记忆池、题目和顺序相同。共享池包含不同任务的事实，属于原作者合成检索压力设置，不是单一真实用户历史；Mem0 可能把这些事实作为冲突或更新处理，这是该设置的限制。125题包含开发域，因此必须单列61题未见域成绩，不把125题当独立测试。

其余子集保留原作者 LUCid test240题和 ImplexConv test81题，候选池不变、分别隔离建库。剩余时间继续这些子集，不启动它们的全量。

Mem0会压缩和合并事实，RootMem生成抽象规则，因此不能为它们伪造原始行Hit@6。共同主指标是相同回答器的隐藏事实利用率：原answer.ANS与JUDGE文本、80词回答、最多6个记忆单元、每单元最多400字符、Qwen输出2048，所有方法一致。RootMem返回激活规则再返回原文，统一限额可能截短规则，必须结合保存的完整矩阵与上下文分析。原始文本四方法另报真实Hit@6。相同Qwen作判分器仍有偏差，需要人工或独立模型后续核验。

## 连续队列

先分别用4条已知事实与迁居更新输入做 smoke 检查，要求正常建库、能取回花生过敏事实、无原生异常；RootMem须有非空规则。检查不是完整复现证明，正式建库和查询仍持续检查。某系统失败时保留错误，其他已验证系统继续，绝不以0分代替失败。

顺序：三系统检查 → InMind同一365条池建库与61题对照 → 通过技术检查的方法继续125题（无论效果好坏）→ ImplexConv与LUCid的相同子集。包括三个记忆系统和plain/HyDE/nojoin/full四个对照。可缓存的模型响应与建库状态保留；每题答案和每项评分落盘。默认7小时窗口，不需要额外确认，不自动根据有利结果选择是否扩展。

原三benchmark全量队列的supervisor临时SIGSTOP，新队列优先；已运行子任务允许完成并保留结果。原队列进度文件可能仍显示其最后任务，需结合previous-queue-paused.json判断。不会在7小时后偷偷恢复旧队列并无限继续消耗。

入口run.sh，status.json是本轮状态；checks/是smoke记录；dataset-audit.json是同题同池审计；states/是原生建库；contexts/是完整取回单元；answers/是逐题答案与判分；results/只写完整成绩；api-responses.jsonl与events.jsonl保存请求、响应和错误。完整覆盖前不报告全量成绩。

## 2026-11-22 16:00 起的主动监测与修复

InMind 原七方法的61题及125题都已完成；RootMem先有8192输出截断，保留旧失败和300条断点，增加构建输出上限到16384后补齐。提示和原生vendor代码未修改。最终矩阵有重复规则，属于生成质量限制，另存诊断；不能把技术完成说成原论文质量完全复现。

Mem0多候选池连续建库时，SDK内部migrations_qdrant目录默认共用并保留客户端锁。现在每池独立sdk-internal目录，显式关闭主客户端及内部客户端；双实例并发打开测试通过。InMind仅一个池，既有有效成绩不受此修复影响。

A-MEM原生代码本就丢弃越界连接提案，并计invalid_link；此前包装器把这项已处理的提案也视为整项失败，过于严格。现保留原生拒绝行为、记录native_rejected_link警告；其他提取、解析、演化错误继续拒收。不得把这些被拒提案宣称为模型输出完全正确。InMind没有该警告，旧分数不变。

空记忆none与只给目标事实的oracle均为原answer.py支持的校准条件，用相同回答器和判分提示补测；oracle含金标准事实，仅为评测校准上界，绝不当作真实检索系统。它们不替换原七方法成绩。修复前源码已按原哈希还原保存到repair-audit/before，修复参数见repair-manifest.json。所有后续新增或修复结果单独落盘。
