# 结果文件说明

主表的逐题回答：`results/native-readout4096-v1/predictions/*.jsonl`。

主表的逐题评分：`results/eval-official-v1/rows/native_readout4096/{answer,grounded}/*.jsonl`，11方法×125题×2指标=2750条。

其它三种历史协议的评分也保留，但不能与主表混合。汇总见`results/eval-official-v1/summary.json`。每条评分带原答案/上下文SHA256；公开预测中的`context_text_sha256`是服务器原上下文的直接UTF-8哈希，而`notes_sha256`类导出占位摘要可能使用JSON序列化，二者不应混淆。

当前补充表和收尾记录：`results/monitoring-20261122/verified-progress-snapshot.json`、`common-prefix10-results.json`、`common-prefix20-results.json`、`final-exit-audit.json`。这些记录有各自的时间戳与固定样本范围。

已完成的ImplexConv81题和LUCid240题简化评分位于`results/other-native-readout4096-v1/results/`。未完整的方法没有相应81/240题成绩。`raw_hit6`只用于原始文本目标行检索，不能给压缩原生记忆强行填同一种命中定义。

原简化判分会把一些一般性谨慎回答当成功；主表改用解释条件的两套提示，但后端仍为本地Qwen。这个变更不是重写或删除旧分数。主表的无记忆、正确事实oracle、写侧/有无属性对照及未见域划分同样必须展示。

具体实验过程、旧数据协议与当前尚未解决的问题分别见同目录的EXPERIMENT_LOG、PROTOCOLS和KNOWN_ISSUES文档。
