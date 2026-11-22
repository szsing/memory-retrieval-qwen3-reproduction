# 核验与重新运行

## 1. 直接复算已完成表格

在Python3.10+下运行：

```bash
python3 scripts/verify_results.py
```

不依赖PyTorch、GPU、API key或原始数据。它从逐题文件重新统计，而不是照抄README数字；验证答案原文哈希、与评分记录的上下文摘要一致、任务ID、125题覆盖及61题划分。输入全文未公开，因此上下文摘要的一致性不是从公开原文再次计算；服务器上2750条原文/评分哈希核验记录另存审计目录。

## 2. 还原实际运行环境

目标环境为Ubuntu22.04、单卡RTX4090。`environment/`分别保存基础、baseline和vLLM环境的pip freeze。不要把三份依赖合并安装；原生Mem0路径要求`mem0ai==0.1.98`。本地OpenAI兼容服务使用vLLM0.11；项目内Qwen tokenizer路径为`models/Qwen3-8B`，模型revision见主README。BGE权重也须提前缓存以满足offline加载。

原始数据应按`manifests/input-files.json`恢复路径和SHA256：

- `rsi-memory/inmind_data/`：官方InMind仓库，包括benchmark数据、背景、manifest；对应三个输入文件哈希在清单中。
- `rsi-memory/bench_data/lucid/lucid.json`：官方MichiganNLP/LUCid的`lucid.json`，不能用`lucid_s.json`替代来声称同一构造。
- `rsi-memory/bench_data/implexconv/implexconv_opposed_trim.json`：原作者使用的trim文件，必须匹配清单哈希；不能静默换成完整opposed parquet。未确认可同字节取得时，应联系作者获取/确认其构造过程。
- `experiments/full-run-v2/vendor/InMind-main/`：官方评测的build_timeline.py与判分提示。三个关键blob与固定提交5a2ab2686d3d5b832575f0288d41ecc93eab358b已核对；`manifests/vendor-files.json`提供服务器依赖文件哈希。
- 原始方法快照分别在`experiments/original-qwen-format-v2/snapshot/`等路径，其输入目录也应指向匹配的数据，而非另一个版本。

常用数据来源：

- https://github.com/imlrz/InMind
- https://huggingface.co/datasets/MichiganNLP/LUCid
- https://huggingface.co/datasets/Kaylee0501/ImplexConv

本发布没有重新执行完整GPU冷启动。输入缺失或哈希不匹配时应停止；不能自动选择“看起来相似”的数据来补齐复现。

## 3. 历史运行链（审阅和恢复指引）

以下说明实际依赖关系，不是一份已通过新环境端到端测试的自动安装脚本。先在新的工作目录恢复代码、数据、环境，设置`MEM_PROJECT_ROOT`为该目录，并使用新的运行截止时间；不要覆盖归档结果。

1. `native-source-aligned-v3/worker.py::prepare`构造源任务，依赖三套源数据。比对dataset-audit和task-and-pool-identity；缺少其一会阻止自动prepare，即使目标只是InMind。
2. `strict241-v1/run_method.py METHOD`完成原生前缀、golden等价核验、逐题追加、检索和旧读取。Mem0/A-Mem/RootMem后续读取需要这些状态与有效响应缓存。
3. `strict241-queryfrozen-v2/run_raw.py METHOD`运行full/hyde/rerank/plain及两个控制。查询特征按原批次固定，不要把旧实验单题生成的结果当成原批次复现。发布的`frozen-query-outputs.json`保留实际模型生成特征和查询摘要；原始查询需从匹配数据按任务顺序重建。
4. 查询冻结目录的三个原生答案显式引用strict241-v1。`native-readout4096-v1/run_method.py METHOD`用于八个方法/消融的完整单元读取；RootMem由`root-native4096-v1/run.py`生成并显式引用；none/oracle引用相同输入的已生成对照。引用不是额外独立样本。
5. `eval-official-v1/rescore.py`对固定答案评分。它的结果与原生成答案分开保存；新工作目录必须建立所需引用，不能误读本公开结果目录为新运行缓存。
6. ImplexConv/LUCid使用`other-native-readout4096-v1/retrieve.py`从V3原生检索接续，再由`evaluate.py`按4096完整单元读取。全量结果只在全部题完整时写入；同题10/20的单独结果不能冒充81题。

## 调度器限制

历史supervisor、priority、repair脚本含当时PID、截止时间和检查点路径。**不得在别人的机器上直接运行这些历史调度器。** 审阅其操作逻辑，使用新的PID/时间/输出目录与明确的任务规格。单独worker仍有完整数据和中间状态依赖，也不是任意入口均可无条件直接执行。

## 精确复现与统计限制

提供的是可审阅的真实服务器快照和可直接复算的公开结果，不承诺模型逐token确定。需要独立冷启动、人工判分、正式时间线协议和更多重复运行，才能形成投稿级结果。审计不通过时不得删除断言、跳过失败、修补半截JSON或把缺失题补零以得到完整表格。
