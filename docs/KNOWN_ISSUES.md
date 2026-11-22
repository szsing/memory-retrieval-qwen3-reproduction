# 已知问题与尚未完成的核验

- 目前主结果不能支持双侧匹配有稳定独立收益，不能仅因完整方法在125题部分指标较高就声称泛化成功。
- 主评分仍是本地Qwen自评，独立人工标注尚未完成。答案应用与grounded的局部不一致不被强行改成一致。
- ImplexConv的简化判分：81题完整方法33、RootMem60；Mem0/A-Mem只回答前20题，不能填入81题排行榜。
- LUCid当前构造不是官方S全量；Mem0只完成3个池检索、RootMem5个、A-Mem无完整池成绩，均没有240题原生回答成绩。
- LUCid RootMem一次长输出有104个技能条目但仅27个不同名称，触及16384上限，拒绝该响应。原请求重试得到5382 tokens、34个不同技能，原始失败未删除。
- 最后一项RootMem长调用在600秒后超时，预算后未再重试。它不进入评分，也不被当成错误答案计零。
- 最终检查161个已完成RootMem矩阵，发现1个LUCid矩阵含同名技能（任务索引1），涉及Technical & Networking Knowledge、Urban & Economic Development、Vehicle & Safety Compliance，各2条。JSON可解析不意味着内容语义无误。未通过去重修改分数；示例见`audit/root-duplicate-output-sample.json`。InMind与ImplexConv已完成矩阵未发现这种完全同名重复。
- 补充运行有1次A-Mem无效链接提议，被原生校验拒绝，无无效边写入。
- BudgetEnd与预算后清理导致的退出码不是模型准确率分数。已保存状态与有效模型响应缓存保留，部分未完成的内存状态可能需要重放，不能承诺零重复工作。
- temperature=0下相同请求也曾返回不同输出；公开哈希确保审计现有记录，不保证GPU冷启动逐token完全相同。
- 本发布保留全部相关历史版本；旧README/源码注释中的“official”“full”或旧分数不能覆盖本仓库当前协议说明。
