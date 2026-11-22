# 上游与许可

原方法来自用户项目 `https://github.com/szsing/memory-retrieval-indirect-requests`，固定提交`d62aa408a4e0b9515eed9313663a025a07b08497`。原方法源码与实验适配分开保留；本项目不把截图中未提供的RootMem组合代码或论文复制分数当成已运行结果。

|组件|上游|固定版本 / 使用方式|
|---|---|---|
|Qwen3-8B|https://huggingface.co/Qwen/Qwen3-8B|b968826d9c46dd6066d109eabc6255188de91218，权重另取|
|BGE-small-en-v1.5|https://huggingface.co/BAAI/bge-small-en-v1.5|同一模型，本地CPU嵌入，权重另取|
|Mem0|https://github.com/mem0ai/mem0|PyPI mem0ai==0.1.98，未复制整个SDK|
|A-Mem|https://github.com/WujiangXu/AgenticMemory|f9a173dec325d84fcc97e7828775c7b241c33466|
|RootMem / IMLogic|https://github.com/ThisIsCosine/IMLogic|8823defc18ec94e377a835695ce9180678eb2b0e，仅RootMem代码，不含IMLogic数据|
|InMind|https://github.com/imlrz/InMind|依赖文件与判分提示哈希保留；三个关键代码/提示blob与5a2ab2686d3d5b832575f0288d41ecc93eab358b匹配|
|LUCid|https://github.com/MichiganNLP/LUCid|092cbf9c95c05377bc0f4a5f644db1dd2b057442，数据另取|
|ImplexConv|https://github.com/Kaylee0501/ImplexConv|202c446bfb1205e981405d44eed3416f52d3b375，数据另取|

打包的A-Mem/RootMem Python文件与固定上游git blob逐个比对，结果见`manifests/vendor-local-changes.json`；本地适配代码位于`experiments/native-source-aligned-v3/native.py`及`experiments/full-run-v2/run_baselines.py`等文件。即使上游文件一致，运行时替换LLM/embedding、存储隔离、预算和读取方式仍属适配，不能宣称原论文全部配置复现。

第三方原有许可证见`third-party-notices/`及对应vendor目录。A-Mem与RootMem代码按其上游MIT声明保留原通知。IMLogic的数据许可与代码不同，本仓库不重新分发该数据。

原始benchmark会话、查询和金标准解释不重新分发；通过官方来源自行获取，并核对`manifests/input-files.json`和候选池身份清单。模型生成回答、判分结果、部分生成状态错误示例及匿名化输入哈希作为实验产物保留。输入全文的本地转储、模型权重、凭证和环境数据库未上传。

原方法来源：[szsing/memory-retrieval-indirect-requests](https://github.com/szsing/memory-retrieval-indirect-requests)，使用提交 `d62aa408a4e0b9515eed9313663a025a07b08497` 的代码快照。原仓库的公开可访问性不作为本发布源码审阅的前提。
