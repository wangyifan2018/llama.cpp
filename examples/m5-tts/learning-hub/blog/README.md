# M5-TTS 端侧推理优化系列

开源仓库：[wangyifan2018/llama.cpp](https://github.com/wangyifan2018/llama.cpp) · [M5-TTS 代码与运行说明](https://github.com/wangyifan2018/llama.cpp/tree/master/examples/m5-tts)

按学习中心原顺序发布时，从[完整版目录（9 篇）](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/README.md)开始。下面四篇适合作为简版。

这组文章以一台无可用 HMX 路径的 Snapdragon 设备为例，记录从模型移植到实时语音播报的工程决策。每篇可独立阅读，实验表格和代码入口集中在[学习中心](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/index.html)与[优化复盘](https://github.com/wangyifan2018/llama.cpp/blob/master/docs/backend/snapdragon/m5-tts-optimization-log.md)。

| 篇目 | 要回答的问题 | 学习中心 |
| --- | --- | --- |
| [01 · 把 M5-TTS 拆给 HTP 和 CPU](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/01-pipeline.md) | 生成一帧音频要跑哪些计算？ | [第 1–3 章](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/01-inference-pipeline.md) |
| [02 · 给一次语音请求做性能剖析](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/02-profiling.md) | RTF 变低时，实际响应真的更快了吗？ | [第 4–5 章](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/03-measure-and-model.md)、[第 10 章](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/06-performance-decisions.md) |
| [03 · KV32、codec 分块和听感回归](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/03-optimization.md) | 哪些改动能在既定音质约束下上线？ | [第 6–9 章](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/04-optimization.md)、[第 11 章](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/07-performance-ceiling.md) |
| [04 · 用 Qwen3-TTS 的 Serena 做座舱播报](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/04-serena.md) | 怎样迁移音色，怎样验证中文导航播报？ | [第 13 章](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/08-transfer-and-voice.md) |

四篇正文可分别粘贴到 CSDN 的 Markdown 编辑器。文中的图均由学习中心现有 SVG 导出，PNG 位于 `images/`。正文已引用 GitHub 上的 PNG。若 CSDN 预览未显示图片，可以上传原 PNG 并保留图注与替代文本。实验链接指向仓库的 master 分支。
