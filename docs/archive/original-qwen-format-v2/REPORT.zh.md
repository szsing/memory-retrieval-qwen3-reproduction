# 原作者脚本＋Qwen3-8B 格式修复版进度

状态：paused_at_budget；当前任务：full-implex-full-fold0

原作者源码与数据校验、原提示/批处理/排序规则均保留。以下仅列已完成任务，不把未完成全量当作全量成绩。

|任务|指标|命中或应用成功|
|---|---|---:|
|full-implex-full-fold0|implex_test|35/79|
|full-implex-hyde-fold0|implex_test|13/79|
|full-implex-plain-fold0|implex_test|18/79|
|full-inmind-full|inmind_off|21/125|
|full-inmind-full|inmind_dense|11/125|
|full-inmind-full|harm|116/125|
|full-inmind-hyde|inmind_off|5/125|
|full-inmind-hyde|inmind_dense|1/125|
|full-inmind-hyde|harm|116/125|
|full-inmind-plain|inmind_off|3/125|
|full-inmind-plain|inmind_dense|2/125|
|full-inmind-plain|harm|110/125|
|full-inmind-rerank|inmind_off|14/125|
|full-inmind-rerank|inmind_dense|6/125|
|full-inmind-rerank|harm|116/125|
|full-lucid-full-fold0|lucid_test|50/240|
|full-lucid-hyde-fold0|lucid_test|43/240|
|full-lucid-plain-fold0|lucid_test|41/240|
|full-lucid-rerank-fold0|lucid_test|45/240|
|subset-implex-full|implex_test|29/81|
|subset-implex-hyde|implex_test|22/81|
|subset-implex-plain|implex_test|15/81|
|subset-implex-rerank|implex_test|25/81|
|subset-inmind-full|inmind_off|13/61|
|subset-inmind-full|inmind_dense|7/61|
|subset-inmind-full|harm|75/85|
|subset-inmind-hyde|inmind_off|5/61|
|subset-inmind-hyde|inmind_dense|2/61|
|subset-inmind-hyde|harm|75/85|
|subset-inmind-legacy_both|testA/dense|5/24|
|subset-inmind-legacy_both|testA/official|17/24|
|subset-inmind-legacy_both|testB/dense|7/61|
|subset-inmind-legacy_both|testB/official|22/61|
|subset-inmind-legacy_both|direct/dense|18/85|
|subset-inmind-legacy_both|direct/official|37/85|
|subset-inmind-legacy_prior|testA/dense|3/24|
|subset-inmind-legacy_prior|testA/official|13/24|
|subset-inmind-legacy_prior|testB/dense|2/61|
|subset-inmind-legacy_prior|testB/official|23/61|
|subset-inmind-legacy_prior|direct/dense|21/85|
|subset-inmind-legacy_prior|direct/official|72/85|
|subset-inmind-legacy_write|testA/dense|4/24|
|subset-inmind-legacy_write|testA/official|11/24|
|subset-inmind-legacy_write|testB/dense|7/61|
|subset-inmind-legacy_write|testB/official|21/61|
|subset-inmind-legacy_write|direct/dense|27/85|
|subset-inmind-legacy_write|direct/official|61/85|
|subset-inmind-plain|inmind_off|1/61|
|subset-inmind-plain|inmind_dense|1/61|
|subset-inmind-plain|harm|73/85|
|subset-inmind-rerank|inmind_off|10/61|
|subset-inmind-rerank|inmind_dense|4/61|
|subset-inmind-rerank|harm|75/85|
|subset-lucid-full|lucid_test|45/240|
|subset-lucid-hyde|lucid_test|40/240|
|subset-lucid-plain|lucid_test|42/240|
|subset-lucid-rerank|lucid_test|42/240|

解析/截断汇总：`{"api_calls": 7167, "truncated": 0, "thinking": 0, "json_attempts": 13716, "invalid_json": 0, "expected_items": 247179, "missing_items": 0, "valid_empty_lists": 136969, "jcall_calls": 12900, "jcall_none": 0}`

自动全量决策：`True`。失败率/截断可能导致不触发；完整原因见alignment-gate.json。

本轮全量指原作者协议下扩大记录/题目覆盖，不等于官方完整长历史协议。
