# M5-TTS 端侧推理实战（09）｜实验手册：复现命令、思考题和术语

开源仓库：[wangyifan2018/llama.cpp](https://github.com/wangyifan2018/llama.cpp) · [M5-TTS 代码与运行说明](https://github.com/wangyifan2018/llama.cpp/tree/master/examples/m5-tts)

汇总可复现的入口，并用习题回顾判断过程。

## 第 14 章 · 实验手册、思考题与术语表

把教程变成可复现的动作。

**本章目标**

- 能照着命令复现本书的关键实验（含回退开关）
- 用分布在各篇中的思考题自测，覆盖全书判断链
- 术语表用于快速对齐口径，避免各说各话

### 14.1 命令速查

``` bash
# 端到端 / 分项统计
curl -s -D - -o out.wav -X POST 127.0.0.1:18080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"今天天气不错，我们一起去公园散步吧。","voice":"serena","response_format":"wav"}' \
  | grep -i x-m5
curl -s '127.0.0.1:18080/metrics?format=prometheus' | grep m5_p50

# 微基准：depth 提交成本模型
adb shell 'cd /data/local/tmp/llama-m5-htp-v73 && \
  LD_LIBRARY_PATH=$PWD ADSP_LIBRARY_PATH=$PWD GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1 \
  ./depth-bench depth-Q14_0.gguf HTP0 7'

# 依赖消融 / codes 逐帧 dump / heads 分项
M5_ZERO_TRANSFORM=1 ./m5-engine model --text "…" --voice 0 --out /tmp/a.wav
./m5-engine model --text "…" --voice 0 --dump-codes /tmp/a.codes --out /tmp/a.wav
M5_HEAD_STATS=1 ./m5-engine model --text "…" --voice 0 --out /tmp/a.wav

# 量化 A/B（同轮交替，避免热漂移）
for i in 1 2; do M5_HEADS_INT8=1 …; M5_HEADS_INT8=0 …; done
./llama-quantize --allow-requantize backbone-Q8_0.gguf backbone-Q14_0.gguf Q14_0

# 一键部署（固定 env、推二进制/控制台、预热、forward）
./deploy.sh
```

### 14.2 综合思考题

**Q11　为什么 7 字句 RTF 0.918 而 24 字句 0.742？这算优化成功吗？**

**不算。**同一份代码，只是固定成本被摊薄。用 `T(N)=截距+斜率×N` 拟合：截距 ≈152 ms（prefill / 图初始化 / codec prime），斜率 ≈54.7 ms/帧（backbone 6.1 + depth 48.6，codec 被重叠）。 短句 23 帧摊 152 ms 固定开销，每帧多背 6.6 ms；长句 75 帧只多 2 ms。所以看性能必须固定输入长度，或同时报告截距与斜率。

**Q12　depth 权重从 21 MB 降到 10.7 MB 只快 9%，你能推出什么？若改成 heads 权重减半呢？**

若瓶颈是权重带宽，字节数减半应接近 2×；只有 9% 说明 **depth 是提交开销 / 计算主导**，继续降位量化没有意义。 而 heads 每帧流式读 34 MB、耗时 13.4 ms ≈ 2.5 GB/s，符合带宽特征；f16→int8 减半后 heads_s 0.94→0.38 s（2.5×），验证了判别。**同一个动作效果相反，因为瓶颈类型不同。**

**Q13　流式首块从 2 帧改成 1 帧，为什么首包反而从 0.285 s 涨到 0.51 s？**

少了一个 depth step 省下的时间，小于 codec 多一次 ORT Run 的固定开销 + 该会话的调度/状态准备成本。 这是 5.1 的 `F` 在 codec 层的翻版：**分块越碎，固定成本占比越高**。结论是维持 2 帧，并把"1 帧更慢"写进代码注释防止后人重试。

**Q14　int8 让 codes 翻转 90%，为什么还能上线？什么情况下你必须否决一个量化？**

首帧 0/16 翻转证明没有系统性数值误差；之后是 `next_hidden` 反馈放大 1-ulp 差异造成的混沌分叉，等价换随机种子。 门禁看 ASR（语义）、音频统计族（分布）、模式内确定性 0% 翻转、人耳。 **否决条件**是感知层面的系统性偏移：backbone Q14 让音频短 16%、语速变快，ASR 全对也不行；或 codec 激活×激活量化掉 1.3 dB。

**Q15　"用 Qwen3-TTS 的音色"这句话里，梯度更新发生在哪里？**

**迁移过程里一次都没有。**Qwen3-TTS 自己的音色能力来自它预训练 / 后训练（含 speaker fine-tuning）时学到的映射； M5 只是把 Qwen 生成的一段音频重新编码成自己词表里的 codec codes，作为 prompt 前缀做 in-context learning。 运行时既不加载 Qwen 也不更新权重。真正"把槽位训练出来"的例子是 Qwen3-TTS 的 VoiceDesign（自然语言描述控制）， M5 的 caption 槽因训练稀疏而弱——这正是"槽位存在 ≠ 槽位可用"。

**Q16　如果让你继续把 RTF 从 0.75 往下压，下一步做什么？**

**先别急着动任何模块——这道题在 2026-11-15 被三项实测改写了（第 10、11 章）。** ① 天花板不是 2.1×：算上 heads 后 depth 上限只有 1.78×，而且 codec 与 depth 并行， `wall ≈ backbone + max(depth, codec) + prefill`；只做草稿会被 codec 卡住。 ② 三条看似可行的路全部排除：单边旋钮与资源让渡（日志 §13，48 次 A/B 无一快于默认）、 codec 单独优化（被 max 吃掉）、学习型草稿（日志 §14，α_oracle ≈ 0.18 ≪ 0.72 盈亏平衡）。 ③ 后续 KV32 和产品分块优化把同一条 69 帧样句的服务端 WAV RTF 降至 0.677（日志 §16、§25）。 下一轮先对当前默认配置重测稳态关键路径；联合重训 depth 和 codec 的收益仍需实验验证（日志 §30）。 这题真正考的是：**先看清依赖结构与并行结构，再判断某个优化能不能兑现。**

### 14.3 术语表

| 术语 | 在本项目里的含义 |
|----|----|
| RTF | Real-Time Factor = 生成耗时 / 音频时长；\<1 表示快于实时 |
| frame / hop | 一帧 = 1920 samples @24 kHz = 80 ms；12.5 fps |
| codebook / book | 每帧 16 个离散码本索引，词表 2051（含 EOS）；M5 的 book 链必须串行采样 |
| prefill / decode | 一次性处理 conditioning 序列 / 逐帧（逐 token）自回归生成 |
| KV cache | llama.cpp 保存的注意力历史；`llama_memory_seq_rm` 可保留指定前缀 |
| Gumbel-style 采样 | `argmax(prob / -log(1-u))`，配合 T=0.9 与 top-k 10；随机数顺序敏感 |
| overlap | codec worker 与 depth 主线程并发，隐藏 codec 墙钟 |
| primer / restore_prime | 参考 codes 最后 8 帧用于推进 codec 递归状态，可在同音色请求间复用 |
| zero-shot voice clone | 不更新权重，靠参考音频的 speaker embedding 或 text–speech pair 上下文迁移音色 |
| MTP | Multi-Token Prediction：Qwen3-TTS 由专用预测器逐步生成每帧剩余码本 |
| dual-track | Qwen3-TTS 文本轨与声学轨并行自回归，收到文本即刻出声学帧 |

### 14.4 延伸阅读

- [`docs/backend/snapdragon/m5-tts-optimization-log.md`](https://github.com/wangyifan2018/llama.cpp/blob/master/docs/backend/snapdragon/m5-tts-optimization-log.md) —— 本教程的数据主台账（含负结果与复活条件）
- [`docs/backend/snapdragon/m5-tts-port-notes.md`](https://github.com/wangyifan2018/llama.cpp/blob/master/docs/backend/snapdragon/m5-tts-port-notes.md) —— 按时间线的完整实验日志
- [`examples/m5-tts/m5_engine.cpp`](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/m5_engine.cpp) / [`m5_depth_heads.h`](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/m5_depth_heads.h) / [`backbone.cpp`](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/backbone.cpp) —— 流水线、采样、llama.cpp 协议
- [`examples/m5-tts/depth_batch_bench.cpp`](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/depth_batch_bench.cpp) —— 自己复现 F + R 成本模型
- [Qwen3-TTS Technical Report](https://github.com/QwenLM/Qwen3-TTS) —— 12Hz/25Hz tokenizer、dual-track、MTP、VoiceDesign、3 秒克隆
- [`tools/tts/README.md`](https://github.com/wangyifan2018/llama.cpp/blob/master/tools/tts/README.md)、[`conversion/qwen3tts.py`](https://github.com/wangyifan2018/llama.cpp/blob/master/conversion/qwen3tts.py)、[`tools/mtmd/mtmd-helper-gen.cpp`](https://github.com/wangyifan2018/llama.cpp/blob/master/tools/mtmd/mtmd-helper-gen.cpp) —— llama.cpp 原生 Qwen3-TTS 实现
- [`examples/m5-tts/README.md`](https://github.com/wangyifan2018/llama.cpp/blob/master/examples/m5-tts/README.md) 的 "Bring a voice from another TTS model" 章节 —— Serena 迁移的原始 recipe

结课 这个案例最值钱的不是某个 RTF 数字，而是三条判断： **① 先量清楚再动手；② 依赖结构决定优化天花板，工程技巧不能突破它；③ 生成式系统的正确性要有感知层验收。** 把这三句带到下一个模型、下一个后端，教程就没白写。
