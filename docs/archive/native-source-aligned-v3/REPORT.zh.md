# 原生记忆系统同模型对照进度

2026-11-22 18:12:34

阶段：remaining_subsets；当前：{'stage': 'retrieve', 'bed': 'implex', 'method': 'rootmem'}

只列完整评测；失败不是0分。InMind使用365条共享池；125题含开发题，61题单列。主指标为相同Qwen回答器的隐藏事实利用率，不能与原始行Hit@6混用。

|数据|方法|已完成题数|利用隐藏事实|
|---|---|---:|---:|
|inmind|amem|125|28/125|
|inmind|amem|61|18/61|
|inmind|full|125|60/125|
|inmind|full|61|27/61|
|inmind|hyde|125|58/125|
|inmind|hyde|61|27/61|
|inmind|mem0|125|38/125|
|inmind|mem0|61|19/61|
|inmind|none|125|38/125|
|inmind|none|61|16/61|
|inmind|oracle|125|110/125|
|inmind|oracle|61|52/61|
|inmind|plain|125|45/125|
|inmind|plain|61|18/61|
|inmind|rerank|125|57/125|
|inmind|rerank|61|24/61|
|inmind|rootmem|125|61/125|
|inmind|rootmem|61|37/61|

解析记录：`{"calls": 9100, "truncated": 3, "thinking": 0, "rejected_attempts": 3, "failed_jobs": 3}`

当前检查和建库不是完整复现结论；以checks/、原生状态、逐题上下文及完整结果共同判断。
