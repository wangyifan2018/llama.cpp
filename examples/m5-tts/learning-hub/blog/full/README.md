# M5-TTS 端侧推理实战 · 学习中心完整版

开源仓库：[wangyifan2018/llama.cpp](https://github.com/wangyifan2018/llama.cpp) · [M5-TTS 代码与运行说明](https://github.com/wangyifan2018/llama.cpp/tree/master/examples/m5-tts)

[交互学习中心 HTML 源文件](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/index.html)原有 15 章。这个发布版本沿用其章节编号和顺序，将相邻主题组织成 9 篇。按顺序读可以从模型结构走到测量、优化、上线和迁移。正文中的「第 N 章」沿用学习中心编号，「日志 §N」指[实验复盘](https://github.com/wangyifan2018/llama.cpp/blob/master/docs/backend/snapdragon/m5-tts-optimization-log.md)。

| 篇次 | 学习中心章节 | 阅读内容 |
| --- | --- | --- |
| [01 · 认识 M5-TTS](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/01-inference-pipeline.md) | 0–1 | 学习路线、文本和参考音色、一帧音频的生成过程 |
| [02 · 从 llama.cpp 到 Hexagon HTP](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/02-llama-cpp-htp.md) | 2–3 | 一次 decode、GGML 图、KV cache、FastRPC、HTP 与 CPU 的计算边界 |
| [03 · 先测量，再建模](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/03-measure-and-model.md) | 4–5 | RTF 与质量门禁、depth 提交成本、合批反例 |
| [04 · 优化复盘](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/04-optimization.md) | 6–7 | 量化、数据搬运、重叠和前缀复用，以及完整的尝试清单 |
| [05 · 上线之后](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/05-production-and-concurrency.md) | 8–9 | 产品分块、热漂移、单请求和双请求的墙钟与 CPU 成本 |
| [06 · 优化决策](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/06-performance-decisions.md) | 10 | 同轮 A/B、等价性校验、停止投入的条件、首包缓存案例 |
| [07 · 重新计算性能余量](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/07-performance-ceiling.md) | 11 | KV32、关键路径、首包选择、联合优化收益估算 |
| [08 · 方法迁移与 Serena](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/08-transfer-and-voice.md) | 12–13 | 通用瓶颈定位方法、Qwen3-TTS 参考音色和 M5 条件生成 |
| [09 · 实验手册](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/learning-hub/blog/full/09-lab-handbook.md) | 14 | 复现命令、综合问答、术语和参考资料 |

## 发布到 CSDN

每篇的一级标题填入标题栏，其余内容粘贴到 Markdown 正文。图片已从学习中心的 SVG 渲染为 PNG。正文已引用 GitHub 上的 PNG。若 CSDN 预览未显示图片，上传对应 PNG 并使用编辑器生成的图片地址。图号沿用学习中心，方便找到原图。每篇发布后可用实际文章 URL 补上系列导航。

图 1-1、5-1、11-2、11-5、13-1 复用[随短稿导出的图片](https://github.com/wangyifan2018/llama.cpp/tree/master/examples/m5-tts/learning-hub/blog/images)。其余 PNG 在本目录的 [images](https://github.com/wangyifan2018/llama.cpp/tree/master/examples/m5-tts/learning-hub/blog/full/images) 中。原有折叠问答在 Markdown 中直接展开，滑块计算器改为固定参数、公式与复算示例。
