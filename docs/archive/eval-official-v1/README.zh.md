# 官方提示的本地复评分

源提示已核对公开上游：

- https://github.com/imlrz/InMind/blob/main/evaluation/prompts/judge_answer_only.txt
- https://github.com/imlrz/InMind/blob/main/evaluation/prompts/judge_application.txt

提示文本不改。user_message、query、explanation来自冻结原始数据；answer与context来自已实际生成、保存的答案和读者可见记忆。金标准解释仅用于判分，不进入记忆构建、检索或回答。后端仍为同一本地Qwen3-8B，temperature0、max_tokens2048、关闭thinking，通过API约束JSON格式。不是原论文模型的官方榜单成绩。

answer指标检查答案是否落实金标准解释；grounded指标还要求读者上下文中存在相应个人信息。generic咨询专业人士等笼统建议不能单独算具体应用。none与oracle为校准条件，oracle用了金标准事实，不能当真实检索方法。

已生成答案不改写；旧简化JUDGE成绩不覆盖。rows/按协议、指标、方法、题号保存，记录答案及上下文SHA256。summary.json仅对完成61/125题的集合给出对应整集成绩，部分完成不充当全量。空上下文grounded非零会记录invariant错误，不强行改成0。

共享365池、逐题241池、241池且冻结原查询特征三套协议分开报告。后一套中输入未变化的原生系统与校准条件显式引用已完成的真实生成记录。不得把缓存或引用当作新运行，也不得跨协议混合分母。
