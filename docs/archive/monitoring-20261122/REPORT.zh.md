# 实验监测记录

2026-11-22 18:49:55

当前主要核验：逐题241池、固定原查询特征、保留原生结构的统一4096-token读取预算、官方提示本地复评分。原共享365池的原脚本分数保留。125题含开发域，61题单列。

## 官方答案应用复评分（仅列完整覆盖）

|协议|方法|61题|125题|
|---|---|---:|---:|
|native_readout4096|full|3|15|
|native_readout4096|write_only_attr|4|11|
|native_readout4096|fact_presence|5|13|
|native_readout4096|rootmem|4|16|
|native_readout4096|mem0|4|11|
|native_readout4096|amem|5|13|
|native_readout4096|hyde|3|11|
|native_readout4096|rerank|4|11|
|native_readout4096|plain|0|6|
|native_readout4096|none|3|12|
|native_readout4096|oracle|44|89|
|shared365|full|4|18|
|shared365|rootmem|11|25|
|shared365|mem0|3|13|
|shared365|amem|3|8|
|shared365|hyde|3|9|
|shared365|rerank|5|14|
|shared365|plain|1|7|
|shared365|none|3|12|
|shared365|oracle|44|89|
|isolated241|full|6|15|
|isolated241|rootmem|5|17|
|isolated241|mem0|4|11|
|isolated241|amem|4|12|
|isolated241|hyde|2|12|
|isolated241|rerank|5|12|
|isolated241|plain|0|7|
|isolated241_frozen_queries|full|3|15|
|isolated241_frozen_queries|write_only_attr|5|13|
|isolated241_frozen_queries|fact_presence|5|12|
|isolated241_frozen_queries|rootmem|5|17|
|isolated241_frozen_queries|mem0|4|11|
|isolated241_frozen_queries|amem|4|12|
|isolated241_frozen_queries|hyde|2|10|
|isolated241_frozen_queries|rerank|4|11|
|isolated241_frozen_queries|plain|0|7|
|isolated241_frozen_queries|none|3|12|
|isolated241_frozen_queries|oracle|44|89|

## 当前生成覆盖

|协议|方法|回答数|旧口径评分数|
|---|---|---:|---:|
|native_readout4096|rootmem|125|0|
|native_readout4096|none|125|125|
|native_readout4096|oracle|125|125|
|native_readout4096|mem0|125|0|
|native_readout4096|full|125|0|
|native_readout4096|amem|125|0|
|native_readout4096|hyde|125|0|
|native_readout4096|rerank|125|0|
|native_readout4096|plain|125|0|
|native_readout4096|write_only_attr|125|0|
|native_readout4096|fact_presence|125|0|
|shared365|mem0|125|125|
|shared365|full|125|125|
|shared365|amem|125|125|
|shared365|plain|125|125|
|shared365|hyde|125|125|
|shared365|rerank|125|125|
|shared365|rootmem|125|125|
|shared365|none|125|125|
|shared365|oracle|125|125|
|isolated241|full|125|125|
|isolated241|mem0|125|125|
|isolated241|amem|125|125|
|isolated241|hyde|125|125|
|isolated241|rootmem|125|125|
|isolated241|rerank|125|125|
|isolated241|plain|125|125|
|isolated241_frozen_queries|mem0|125|125|
|isolated241_frozen_queries|amem|125|125|
|isolated241_frozen_queries|rootmem|125|125|
|isolated241_frozen_queries|none|125|125|
|isolated241_frozen_queries|oracle|125|125|
|isolated241_frozen_queries|hyde|125|125|
|isolated241_frozen_queries|rerank|125|125|
|isolated241_frozen_queries|full|125|125|
|isolated241_frozen_queries|plain|125|125|
|isolated241_frozen_queries|write_only_attr|125|125|
|isolated241_frozen_queries|fact_presence|125|125|

注意：官方提示仍由本地Qwen判分；oracle使用金标准事实，非真实检索系统。grounded还要求上下文包含相应事实，结果另存status.json。旧简化判分存在偏宽，不能据此直接证明方法有效。

该记录不覆盖或重写任何已完成成绩。
