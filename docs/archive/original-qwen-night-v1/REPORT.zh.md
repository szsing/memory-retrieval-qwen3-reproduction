# 原作者脚本＋Qwen3-8B 夜间进度

状态：subset_finished_full_not_triggered；当前任务：subset-answer-implex-rerank

原作者源码与数据校验、原提示/批处理/排序规则均保留。以下仅列已完成任务，不把未完成全量当作全量成绩。

|任务|指标|命中或应用成功|
|---|---|---:|
|subset-answer-implex-full|implex_val/answer|0/79|
|subset-answer-implex-hyde|implex_val/answer|0/79|
|subset-answer-implex-plain|implex_val/answer|0/79|
|subset-answer-implex-rerank|implex_val/answer|0/79|
|subset-answer-inmind-full|inmind/answer|7/61|
|subset-answer-inmind-hyde|inmind/answer|6/61|
|subset-answer-inmind-plain|inmind/answer|0/61|
|subset-answer-inmind-rerank|inmind/answer|12/61|
|subset-implex-full|implex_test|25/81|
|subset-implex-hyde|implex_test|22/81|
|subset-implex-plain|implex_test|15/81|
|subset-implex-rerank|implex_test|26/81|
|subset-inmind-full|inmind_off|10/61|
|subset-inmind-full|inmind_dense|5/61|
|subset-inmind-full|harm|76/85|
|subset-inmind-hyde|inmind_off|4/61|
|subset-inmind-hyde|inmind_dense|2/61|
|subset-inmind-hyde|harm|76/85|
|subset-inmind-legacy_both|testA/dense|3/24|
|subset-inmind-legacy_both|testA/official|9/24|
|subset-inmind-legacy_both|testB/dense|9/61|
|subset-inmind-legacy_both|testB/official|34/61|
|subset-inmind-legacy_both|direct/dense|19/85|
|subset-inmind-legacy_both|direct/official|44/85|
|subset-inmind-legacy_prior|testA/dense|0/24|
|subset-inmind-legacy_prior|testA/official|12/24|
|subset-inmind-legacy_prior|testB/dense|5/61|
|subset-inmind-legacy_prior|testB/official|29/61|
|subset-inmind-legacy_prior|direct/dense|20/85|
|subset-inmind-legacy_prior|direct/official|74/85|
|subset-inmind-legacy_write|testA/dense|3/24|
|subset-inmind-legacy_write|testA/official|11/24|
|subset-inmind-legacy_write|testB/dense|9/61|
|subset-inmind-legacy_write|testB/official|37/61|
|subset-inmind-legacy_write|direct/dense|34/85|
|subset-inmind-legacy_write|direct/official|74/85|
|subset-inmind-plain|inmind_off|1/61|
|subset-inmind-plain|inmind_dense|1/61|
|subset-inmind-plain|harm|73/85|
|subset-inmind-rerank|inmind_off|6/61|
|subset-inmind-rerank|inmind_dense|4/61|
|subset-inmind-rerank|harm|76/85|
|subset-lucid-full|lucid_test|53/240|
|subset-lucid-hyde|lucid_test|41/240|
|subset-lucid-plain|lucid_test|42/240|
|subset-lucid-rerank|lucid_test|42/240|

解析/截断汇总：`{"api_calls": 1967, "truncated": 0, "thinking": 0, "json_attempts": 2980, "invalid_json": 972, "expected_items": 38182, "missing_items": 26869, "valid_empty_lists": 6110, "jcall_calls": 2259, "jcall_none": 377}`

自动全量决策：`False`。失败率/截断可能导致不触发；完整原因见alignment-gate.json。

本轮全量指原作者协议下扩大记录/题目覆盖，不等于官方完整长历史协议。
