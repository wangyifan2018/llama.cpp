# Qwen3-TTS-0.6B 在 Snapdragon HTP 上的流式推理设计

> **2026-11-20 实测更新**：本文是基于算账的设计推演，其中若干结论已被实测推翻，
> 以 [qwen3-tts-opt-log.md](qwen3-tts-opt-log.md) 为准。主要差异：
> 1. **HTP 的价值**：talker 在 HTP 上慢 3.7x（52 vs 14 ms），vocoder 在 HTP 上与 CPU 持平（~246 vs ~205 ms/帧）；
>    本文 §3.3 假设"HTP 适合大 GEMM/conv"在 v73 上不成立，因为 conv 被降级成 im2col+matmul。
> 2. **瓶颈位置**：不是 predictor 的字节数，而是 **vocoder 的算力**（稳态 111 ms/帧，折叠 int8 后）。
> 3. **"卡 3–99 秒"**：不是代码 bug，是 **DVFS 爬频**（前 ~10 秒慢，预热可解）。
> 4. **RTF<1**：需要 P2a（predictor 1 次前向）+ vocoder 减重重训 + 通道收窄，三件套齐了才 ~0.87。

目标：单路流式 TTS，**RTF < 1**，**TTFA 尽量短**，权重与图对 HTP 静态友好。

本文先把模型拆到"每帧要过多少字节、多少算子、多少 MAC"，再给出静态图切分、流式调度与实施阶段。
所有结构数字来自本仓库 [conversion/qwen3tts.py](conversion/qwen3tts.py) 与 [tools/mtmd/models/qwen3tts-gen.cpp](tools/mtmd/models/qwen3tts-gen.cpp)，
配置数字来自 Qwen3-TTS-12Hz-0.6B-Base 的 `config.json` 与 `speech_tokenizer/config.json`。

真机锚点（本仓库已测）：
- **SM6650** = Hexagon v73，2xHVX 128B，**无 HMX**，2 MB VTCM，HVX 数据通路 26 GB/s（[m5-tts-port-notes.md:341](m5-tts-port-notes.md)）。
- **v79 级旗舰** = HMX 12 TFLOPS fp16，8 MB VTCM，DMA 60 GB/s（arXiv:2509.23324）。
- ggml-hexagon 是 **op-by-op 下发**，每算子约 100–137 µs（[m5-tts-port-notes.md:400-413](m5-tts-port-notes.md)）。

---

## 0. 一页结论

1. **这不是一个 0.6B 模型，是 1.03B 的三段式**：talker 601M + code_predictor 143M + code2wav 118M（+ 311M 文本嵌入，只在 prefill 用）。
2. **两个瓶颈互不相干，必须分开打**：
   - **字节账**：每帧 1850 MB = 23 GB/s，是 v73 的 2.9–5.1 倍。**predictor 占 73%**（16 次前向各读 82 MB）。
   - **算力账**：每帧 3.01 GMAC = 37.6 GMAC/s。**vocoder 占 82%**（2.48 GMAC/帧，94% 在 4 个 DAC block 的残差单元）。
   只修 predictor，RTF 从 3.8 降到 1.02 就停住；只修 vocoder，wall 仍被 LM 的 301 ms 钉死。**必须两条腿一起走。**
3. **静态图能消掉提交税（137 µs/算子），但既消不掉权重流量，也消不掉 MAC。**
   0.6B@12.5 Hz 的字节数由 predictor 的自回归结构决定，算力由 vocoder 的结构决定 —— 两端都在算法侧，不在部署侧。
4. **模型本身对流式极友好**：全因果、**零 lookahead**、无 diffusion/flow 迭代、每帧固定 16 codes、vocoder 滑窗 72 帧且有显式 conv/KV 状态。
   流式不需要改结构，只需要改调度（现在 `window_frames = 72` 意味着 5.76 s 才出第一包）。
   参考实现本来就原生流式（`non_streaming_mode=False` 是默认，文本按帧喂，官方称 1 个字符即可出包、97 ms）；
   **本仓库实现的是非流式**（整段文本进 prefill），长文本在 v73 上 TTFA 会到 1-2 s。这是 P0 里最便宜、收益最大的一项。
5. **落地阶梯（目标机型已定 v73 = SM6650）**：

| 阶段 | 改动 | LM ms | voc ms | wall | RTF |
|---|---|---:|---:|---:|---:|
| 现状 | — | 301 | 82 | 301 | **3.8** |
| P0 | 流式化 + 静态图 + 分块（不改模型） | 301 | 82 | 301 | 3.8（但 TTFA 5.8 s -> ~1.2 s） |
| P1 | LM 4 bit | 150 | 82 | 150 | 1.88 |
| P2a | **predictor 1 次前向（重训）** | 53 | 82 | 82 | 1.02 |
| P2b | **vocoder int8 + 减重（重训）** | 53 | 37 | 53 | **0.66** |
| P3 | QNN context binary + DLBC | ~45 | 37 | ~45 | ~0.56 |

6. **v73 的现实结论**：RTF ≈ **0.66–0.74**（P2a+P2b 都做完，约 25–35% 余量），TTFA ≈ **130–160 ms**。
   只做 P2a 会卡在 1.02，只做 P2b 会卡在 1.88。**这是一个"两边都要改模型"的项目，不是一个部署调优项目。**

**一句话**：先改 predictor 的自回归结构（字节），再改 vocoder 的残差单元（算力），静态图只在中间负责把 2050 个算子揉成 1 次提交 —— 三步缺一不可。

---

## 1. 模型解剖

### 1.1 三段式结构

```
文本 ──► [speaker encoder ECAPA-TDNN] ──┐ （可选，参考音色，每请求一次）
                                     ▼
文本 token ──► text_projection(2048->1024，转换期折进 embedding 表)
                                     │
                                     ▼
  1) talker  Qwen3 0.6B, 28 层, d=1024, 16Q/8KV x 128, FFN 3072   -> 1 token / 帧
       输入 = Σ(16 个 codebook embedding) + 1 行 text hidden
              （流式：该帧对齐的文本行；非流式：tts_pad）
       输出 = semantic code0 + hidden[1024]
                                     │
                                     ▼
  2) code_predictor  5 层, d=1024, 16Q/8KV x 128, FFN 3072         -> 16 次前向 / 帧
       位置 0  = project_in(talker hidden)，只填 KV
       位置 1  = embd(code0) -> lm_head[0] -> sample code1
       ...     位置 15 -> sample code15（严格自回归，逐 code 串行）
                                     │
                                     ▼
  3) code2wav  8 层 transformer(d=512, 16x64, FFN 1024, 滑窗 72)   -> N 帧 / chunk
       RVQ decode(16 books) -> pre_conv(512->1024)
       -> 2x(ConvTranspose x2 + ConvNeXt)
       -> DAC: conv_pre -> 4 block[Snake + CT(8,5,4,3) + 3 res]
       -> Snake -> conv_post -> 24 kHz PCM
```

关键结构常数（全部可从 config 读出，不等同于"常识"）：

| 项 | 值 | 来源 |
|---|---|---|
| 帧率 | **12.5 Hz**（80 ms/帧，`_frame_rate: 12.5`） | speech_tokenizer/config.json |
| 上采样倍数 | 1920 = 2x2 x **8x5x4x3** -> 24 kHz | `decode_upsample_rate` |
| 每帧 code 数 | 16（1 semantic + 15 acoustic） | `num_code_groups: 16` |
| codebook | semantic 4096 / acoustic 2048 x 15 | `semantic_codebook_size` |
| talker | 28 层, h=1024, 16Q/8KV x 128, FF=3072 | talker_config |
| code_predictor | **5 层**, h=1024, 16Q/8KV x 128, FF=3072 | code_predictor_config |
| code2wav | 8 层, h=512, 16Q/16KV x 64, FF=1024, **滑窗 72 帧** | decoder_config |
| talker vocab | text 151936 / codec 3072 | talker_config |

### 1.2 参数量与每帧字节（int8，含 3% per-block scale）

| 模块 | 结构 | 参数 | 每帧读权重 | 每帧 MAC |
|---|---|---:|---:|---:|
| talker core | 28 x (q,k,v,o + gate,up,down) | 440.5 M | 456 MB | 442 M |
| talker embed | text 151936x**2048** + codec 3072x1024 | 311.2 M + 3.1 M | **只在 prefill 用**（CPU gather） | - |
| talker head | 1024 x 3072 | 3.1 M | 3 MB | 3 M |
| **code_predictor** | 5 层 x 16 次前向 + 15 个 lm_head | 142.6 M | **1348 MB** | **1310 M** |
| code2wav | 8 层 tfm + RVQ + 2x up + 4 个 DAC block | 118 M | 31 MB（N=4 摊薄） | **2.48 G** |
| **合计 / 帧** | | **~1.02 B** | **1850 MB** | **3.01 G** |

换算：**23.1 GB/s 持续带宽、37.6 GMAC/s、约 2050 个算子/帧**。
（KV 另计：talker 57 KB/帧位置，256 帧上下文 int8 约 15 MB/帧，占比 <1%。）

**注意这两列的分布完全不同**：字节由 predictor 主导（73%），算力由 vocoder 主导（82%）。
这是两条独立的战线，不存在"改一处两边都降"的方案 —— 也是本文后面要分 P2a / P2b 的原因。
（官方参数量：base 914.6 M + tokenizer decoder 118 M ≈ 1.03 B，与上表一致。）

### 1.3 结构性质：哪些对 HTP/流式是利好

| 性质 | 影响 |
|---|---|
| **全因果，零 lookahead** | vocoder 可以 1 帧粒度出流，不需要未来帧；不像 flow-matching vocoder 要跑 K 步。这是本模型最大的结构红利。 |
| **每帧固定 16 codes，固定帧率** | 所有 shape 可静态化，天然适合 context binary。 |
| **vocoder 是 8 层 attention(滑窗 72) + 一堆 causal conv** | 无 diffusion 迭代、无 CFG、无 solver 步数，算子数固定，不用调采样步数。 |
| **vocoder 已带显式 state（conv 左上下文 + KV 尾）** | 分块解码与整段解码等价（左上下文 >=72 帧时严格等价；官方默认 25 帧，属可接受近似），已有实现（`list_c2w_state_slots`）。 |
| **talker 每帧只 1 token** | batch=1 GEMV，HMX 上只填 1/32 行（[arXiv:2509.23324](https://arxiv.org/html/2509.23324v1)）；这是 decode 效率的结构性上限。 |
| **predictor 严格自回归 15 步** | 每步依赖上一步"采样出的 code"，无法并行、无法投机（实测接受率 0–1.5%）。**唯一的坏性质**。 |

### 1.4 参考实现原生的流式形态（Dual-Track）

官方模型卡与参考源码里的事实，直接决定了 pipeline 该怎么搭：

| 事实 | 出处 | 对本设计的意义 |
|---|---|---|
| `non_streaming_mode=False` 是**默认值** | modeling_qwen3_tts.py:2026 | **流式才是参考的默认路径**，本仓库实现的是非流式 |
| 流式下 talker 每帧输入 = Σ(16 code embedding) + **该帧对齐的 1 个 text hidden** | modeling:1687-1694, 2225 | 文本按帧"喂"，prefill 只含固定前缀 9-10 行 |
| 非流式下整段文本进 prefill，每帧只加 `tts_pad` | modeling:2205-2231 | 本仓库当前实现（[mtmd-helper-gen.cpp:253](tools/mtmd/mtmd-helper-gen.cpp)） |
| 官方："输入 1 个字符即可出首个音频包，端到端延迟 97 ms" | README:54 | TTFA 目标量级（云端 GPU） |
| vocoder 是 causal + `sliding_window=72`，官方自己用 `chunked_decode(chunk_size=300, left_context=25)` | tokenizer_12hz:291, 886-894 | 分块解码是官方路径；**25 帧左上下文就够**，不必留 68 帧 |
| 采样默认 `temperature=0.9, top_k=50, repetition_penalty=1.05, min_new_tokens=2` | modeling:2035-2058 | 本仓库 `do_sampling` 只做 top_k/top_p（无温度、无重复惩罚），韵律会漂 |

两条直接结论：

1. **HTP 上必须实现流式模式**。非流式的 prefill 与文本长度成正比，100 字文本在 v73 上就是 1-2 s 的 TTFA（见 §5.2）；流式的 prefill 是常数。
2. **vocoder 的左上下文可从 68 帧收到 25-32 帧**，state 从 4.7 MB 降到 1.8 MB，且与官方默认口径一致。

---

## 2. 每帧预算：四本账（v73）

80 ms 一帧，四个硬上限。**关键是要分清"谁卡在哪个账上"**：

| 账 | v73 预算 | 现结构 | 谁超支 |
|---|---:|---:|---|
| 有效带宽（4.5–8 GB/s） | 360–640 MB | 1850 MB | **predictor**（1348 MB，73%） |
| 算子数（100–137 µs/个） | ~650 个 | ~2050 | **predictor**（800 个，80%） |
| MAC（HTP 30 GMAC/s） | 2.4 G | 1.78 G（LM 侧） | 不超 |
| **MAC（CPU，vocoder）** | **2.4 G** | **2.48 G** | **vocoder 本身就在 1.0 RTF** |

读法：

1. **LM 侧是"字节 + 算子数"问题**，唯一解是把 predictor 从 16 次前向压成 1 次（P2a）。
2. **vocoder 侧是"纯算力"问题**：2.48 GMAC/帧、31 GMAC/s，94% 花在 4 个 DAC block 的残差单元上。
   它和 LM 不共享瓶颈，所以**必须并行**（LM 上 HTP、vocoder 上 CPU），而不是排队。
3. 两边都优化完，wall = max(LM_时间, vocoder_时间) ≈ 53–82 ms -> **RTF 0.66–1.02**。
   只做 LM 不做 vocoder，wall 被 vocoder 钉在 82 ms，**RTF 兜在 1.0 附近，做了等于没做**。

### 2.1 LM 侧：字节账 -> 时间

```
LM_time = 每帧字节 x 12.5 / 有效带宽        （v73: 4.5–8 GB/s）
```

| 配置 | LM 字节/帧 | GB/s@12.5 | ms@4.5 | **@6** | @8 | 评价 |
|---|---:|---:|---:|---:|---:|---|
| A 现状（16 pass, W8） | 1805 MB | 22.6 | 401 | **301** | 226 | 不可行 |
| B 16 pass, W4 | 902 MB | 11.3 | 200 | **150** | 113 | **4 bit 救不了结构** |
| C **1-pass, W8** | 549 MB | 6.9 | 122 | **92** | 69 | 只靠结构改造 |
| D 1-pass, 6 bit talker | 434 MB | 5.4 | 96 | **72** | 54 | 刚好 1.0 |
| **E 1-pass, 4 bit talker** | **320 MB** | **4.0** | **71** | **53** | **40** | **推荐** |
| F 8 book（仍 8 pass）W4 | 565 MB | 7.1 | 126 | **94** | 71 | 不改模型的最好档 |

读表：

- **不改结构，4 bit 也进不了实时**（B = 150 ms）。顺序必须是"先结构、后位宽"。
- 结构改造（C）把 301 ms 压到 92 ms，**仍差一口气**；6 bit（D）72 ms；4 bit（E）53 ms 才有余量。
- **F 是"不改模型"能拿到的最好结果**（只喂前 8 个 codebook，后 8 个补 0）：94 ms，**RTF 仍 1.27**。
  这就是"不重训 = 做不到实时"的量化证据。

### 2.2 全链路：wall = max(LM, vocoder)

vocoder 的 2.48 GMAC/帧跟 LM 不共享瓶颈，只能并行（LM 上 HTP、vocoder 上 CPU）：

| 配置 | LM ms | voc ms（未优化 / int8 / int8+减重） | wall | RTF |
|---|---:|---:|---:|---:|
| A 现状 | 301 | 82 / 55 / 37 | 301 | 3.8 |
| C 1-pass W8 | 92 | 82 | 92 | 1.15 |
| D 1-pass 6bit | 72 | 82 | 82 | 1.02 |
| **E 1-pass 4bit** | **53** | **82 / 55 / 37** | **82 / 55 / 53** | **1.02 / 0.69 / 0.66** |
| E + P2b（int8 + res unit 减重） | 53 | 37 | 53 | **0.66** |

（脚本 [qwen3-tts-roofline.py](qwen3-tts-roofline.py) 会把 KV 与 vocoder 权重也计入字节项，输出 wall 59 ms / RTF 0.74；
差异来自那 45 MB，不影响结论。取 0.66–0.74 这个区间即可。）

**这是本文最重要的表**：只做 LM，wall 被 vocoder 钉在 82 ms，RTF 兜在 1.0 —— **做了等于没做**。
必须 P2a（LM）+ P2b（vocoder）同时做，才能稳到 0.66 并留 25% 余量。
（@4.5 GB/s 时 E 的 LM 变 71 ms -> wall = max(71, 37) = 71 ms，RTF 0.89；若 vocoder 没做 P2b，wall = max(71, 82) = 82，RTF 1.02。带宽实测值决定生死，见 §8。）

### 2.3 为什么 vocoder 是 82% 的 MAC

| 子模块 | 速率 | GMAC/s | 占比 |
|---|---:|---:|---:|
| pre_transformer 8L | 12.5 Hz | 0.37 | 1.2% |
| upsample #0 / #1 | 25 / 50 Hz | 0.26 / 0.52 | 2.5% |
| DAC entry (1024->1536, k7) | 50 Hz | 0.55 | 1.8% |
| DAC block s=8 -> 400 Hz, ch768 | 400 Hz | 6.61 | 21.3% |
| DAC block s=5 -> 2 kHz, ch384 | 2 kHz | 8.26 | 26.6% |
| DAC block s=4 -> 8 kHz, ch192 | 8 kHz | 8.26 | 26.6% |
| DAC block s=3 -> 24 kHz, ch96 | 24 kHz | 6.19 | 20.0% |
| post conv (96->1, k7) | 24 kHz | 0.02 | 0.1% |
| **合计** | | **31.0** | = 2.48 GMAC/帧 |

**94% 在 4 个 DAC block 的残差单元**（每 block 3 个：k=7 空洞卷积 + k=1 卷积），
因为它们要在 400 Hz–24 kHz 上跑 768/384/192/96 通道。这部分是"每个输出样本都要乘一遍权重"，
**不随 chunk 摊薄** —— 所以"vocoder 权重只占 2% 字节"骗不了人：它是纯算力题。

三条结论：

1. vocoder 必须 int8 + Snake 多项式/LUT + CT->MatMul（M5 已验证这套值 -60%，codec 5.28 s -> 2.01 s）。
2. 想再降 20%+ 只能动结构（res unit 3->2、通道收窄、最后一级降通道），属 P2b 重训范畴。
3. **不要**把 vocoder 搬到 v73 的 HTP：无 HMX，31 GMAC/s 已超过整颗 DSP 的实测吞吐（10.6 GMAC/s）。

### 2.4 为什么 predictor 是 73% 的字节

| 模块 | 每帧字节 | 占字节 | 每帧 MAC | 占 MAC |
|---|---:|---:|---:|---:|
| talker（1 次 28 层） | 456 MB | 25% | 442 M | 15% |
| **code_predictor（16 次 5 层）** | **1348 MB** | **73%** | 1310 M | 43% |
| **code2wav（N=4 摊薄）** | 31 MB | 2% | **2480 M** | **82%** |

predictor 只有 78.6 M 参数的 core，却被读 16 遍。**它的算术强度是 1 MAC/byte，而 HTP 需要 10–30 MAC/byte 才不浪费。**
对比：同一个 5 层模型如果一次跑 16 行（像 prefill 那样），算术强度变 16，字节还是 82 MB，时间降到 1/2.1
（本仓库 M5 实测 1x28 vs 28x1 差 2.1 倍，[m5-tts-optimization-log.md:88](m5-tts-optimization-log.md)）。

**所以 predictor 的正确形态是"一次前向出 16 个 code"，而不是 16 次前向各出 1 个。**
这需要改模型头；规格、训练配方与代码改动点见 [qwen3-tts-predictor-redesign.md](qwen3-tts-predictor-redesign.md)。
顺带一个结构红利：单位置注意力会退化成"恒等作用于 V"，q/k/softmax 可以整个删掉，5 层 transformer 塌成 5 层 FFN。

---

## 3. HTP 静态图映射

### 3.1 图清单（每张图 = 一个固定 shape 的常驻图）

| 图 | 输入 | 状态（设备常驻） | 输出 | 触发 |
|---|---|---|---|---|
| **G0 spkenc** | mel[128, T], T∈{256,512,768} | 无 | embd[1024] | 每请求一次（应缓存） |
| **G1 prefill** | embd[1024, L], pos[L], mask[L,L], L∈{32,64} | talker KV（写入） | last hidden[1024] | 请求开始 |
| **G2 frame** | talker_in[1024], pos, kv_len | talker KV[28,2,1024,B] + predictor KV[5,2,1024,16] | codes[16] i32, hidden[1024] | 每帧 |
| **G3 wav[N]** | codes[16, W]，W = 32+N（右补 0） | 8x2x1024x32 KV + 21 个 conv tail | pcm[1920 x N] | 每 chunk，N∈{1,2,4} |

要点：

1. **G2 必须把 talker decode 与 code_predictor 融成一张图。**
   现在是两张图 + 一次 hidden 回传 + 一次 CPU 采样（[mtmd-helper-gen.cpp:320-372](tools/mtmd/mtmd-helper-gen.cpp)）。
   融合后每帧只剩 **1 次提交、0 次同步**：采样在图上做（top-k/top-p 形状固定，[qwen3tts-gen.cpp:6-68](tools/mtmd/models/qwen3tts-gen.cpp) 已经是纯 ggml 实现），
   采出的 code 用 `get_rows` 直接喂回 predictor，全程不回 host。
2. **KV 必须是设备常驻 buffer，不能 past/present 整块回传。**
   28 层 x 2 x 1024 x 512 帧 int8 = 29 MB，每帧回传一次就是 29 MB/帧（=0.36 GB/s 的纯搬运 + 一次同步），不可接受。
   ggml-hexagon 天然是原地读写（attention 直接读 shared buffer），QNN 用 `LLM native KVcache` 或固定 max_len + mask 的图内 state。
3. **code2wav 的 state 目前是 host 回传**（`state_in_*`/`state_out_*`，[qwen3tts-gen.cpp:751-765](tools/mtmd/models/qwen3tts-gen.cpp)），
   每 chunk 约 4.7 MB x 2。`c2w_state_slot` 已有 `in_place` 开关（[models.h:535](tools/mtmd/models/models.h)），但 qwen3-tts 没用。
   流式下必须切到 in_place，否则 N=1 时这 9.4 MB 比 vocoder 本身还贵。
4. **shape 分档而不是动态 shape**：G2 的 talker KV 按 128/256/512 三档；越过档位就换图。
   本仓库 M5 的经验是 pad 128 最优（[m5-tts-optimization-log.md:1506](m5-tts-optimization-log.md)）：越细越省带宽，但换图次数上升。
   TTS 句子通常 < 400 帧，三档足够，且**三张图可以全部预编译常驻**，不存在运行时换图的开销。
5. **算子数预算**：G2 现在是 ~2050 个算子（108 个 layer 实例 x ~19 op/层）。按 100 µs/算子算是 205 ms，光调度就超预算。
   必须做算子合并：`qkv 合并`（3->1）、`norm+mul` 融合、`gate/up+swiglu` 融合、`rope 双路合并`、`flash_attn_ext` 单算子。
   目标 **<= 7 op/层 -> ~760 个**，正好落在 650–800 的预算内。

### 3.2 静态图的三种"甜头"，要分清哪个能用

| 甜头 | 说明 | 对本模型 |
|---|---|---|
| 消提交税 | 一次 FastRPC + 预编译微码，替代 137 µs/算子 | **是最大收益**，尤其对 2050 算子的 G2 |
| 权重预重排/预分块 | repack 成 HMX tiled layout，K 必须 32 对齐 | 必需；1024/3072 都对齐，OK |
| 权重常驻片上 | 不可行 | 600 MB 权重 > 2 MB VTCM，**每帧仍然从 DDR 读**，所以静态图不减少带宽 |

换句话说：**静态图解决的是"每个字节的搬运效率"，不解决"要搬多少字节"。**
本模型的字节数由算法结构决定，这才是要动刀的地方。

### 3.3 两条栈怎么选

| 维度 | ggml-hexagon（本仓库） | QNN context binary |
|---|---|---|
| 执行模型 | op-by-op 解释执行，每次 100–137 µs/op | 整图编译，无逐 op 开销 |
| 2050 算子的 G2 | **~205 ms 纯调度，直接出局** | 一次 execute，可行 |
| 动态 shape | 原生支持（每次传 shape） | 必须固定，需分档 + weight sharing |
| KV | 原地读写，零拷贝 | 需 LLM-KVcache 特性或 max_len+mask |
| 改模型 | 不用重编译 | 必须重新生成 context binary |
| 多档 shape | 图按 uid 缓存，天然 | 每档一个 graph，**必须共享权重**（v73+、单 PD、<=64 graph） |

结论：**原型验证走 ggml-hexagon（改图快），量产/RTF<1 必须走 QNN context binary**（或等 ggml-hexagon 支持整图编译）。
若短期拿不到 QNN SDK，退路是"把 G2 拆成 8 张更小的图"（每 3–4 层一张），把算子数摊到多个提交上，但这会引入 8 次同步，收益有限。

### 3.4 不该上 HTP 的部分

| 部分 | 放哪 | 理由 |
|---|---|---|
| text embedding gather（151936x1024） | **CPU** | 只在 prefill 用，155 MB 表没必要进 DSP 内存；本仓库已经是以 embedding 行喂入（`decode_embd_batch`） |
| speaker encoder | CPU（结果缓存） | 每请求一次，可离线；且它是 conv 小算子，HTP 收益 <1.6x |
| top-k / top-p / cumsum 采样 | 图内（G2）或 CPU | 图内省一次同步；CPU 省算子数。二选一，先按图内做 |
| vocoder | **CPU（3 线程 NEON）** | 只占每帧 2% 字节；上同一颗 DSP 会与 G2 串行（M5 实测搬 DSP 后墙钟 +40%） |
| Snake / sin / exp | 必须多项式或 LUT | HVX 无 sin 快路径，M5 的 `m5_snake_op.h` 9 阶多项式可直接复用 |

---

## 4. 流式流水线

### 4.1 现在为什么不是流式

`window_frames = 72`（[mtmd-helper-gen.cpp:500](tools/mtmd/mtmd-helper-gen.cpp)）：codes 攒够 72 帧才调 vocoder，
即 **5.76 s 才出第一包**；而且 G3 的图固定按 72 帧建（[qwen3tts-gen.cpp:744](tools/mtmd/models/qwen3tts-gen.cpp)），
即使只喂 4 帧，也算 72 帧的活，再按 `n_frames` 裁掉尾巴（[mtmd-subcontext.cpp:2630-2633](tools/mtmd/mtmd-subcontext.cpp)）。
**这是"批处理"实现，不是流式实现。**

### 4.2 chunk 调度：1, 2, 4, 4, 4 ...

左上下文按官方 `chunked_decode(left_context=25)` 取 **25-32 帧**（不要 68）：
vocoder 的 KV state 从 4.7 MB 降到 1.8 MB，attention 行数也少一半以上。
若要用满 72 帧窗口（更"精确"，但与官方默认不同），把 W 改成 72+，代价是 state 与每 chunk 的 attention 读带宽。

| 段 | chunk | 目的 | 代价 |
|---|---|---|---|
| 首包 | **N=1**（80 ms 音频） | TTFA 最小 | vocoder 权重读 120 MB 摊到 1 帧 = 1.5 GB/s（一次性） |
| 第 2 包 | N=2 | 填缓冲 | 0.75 GB/s |
| 稳态 | **N=4**（320 ms） | vocoder 摊薄到 31 MB/帧（2%） | 输出粒度 320 ms，够给 AudioTrack 吃 |

代价对比（v73，5 GB/s）：N=1 常年跑 = 每帧 +24 ms（+30%），N=4 = 每帧 +6 ms（+7%）。
所以**首包用 N=1，稳态必须回到 N=4**。

### 4.3 线程模型

```
   [HTP 线程]                                  [CPU 线程]
   G1 prefill ─┐
               ▼
        ┌─────────────┐  codes[16]   ┌──────────────┐
        │ G2 frame    │─────────────►│ G3 wav[4]    │──► PCM ring ──► AudioTrack
        │ talker+pred │   (ring buf) │ (NEON x3)    │    (400 ms)
        └─────────────┘              └──────────────┘
               │ 每帧 80 ms 时钟                 ▲
               └──────── 与 HTP 并行 ───────────┘
```

- **只在 HTP 上放一张图**（G2）。不要试图让 G2 和 G3 在两个 HTP session 上并行：同一颗 HTP0 是分时复用的，
  双会话实测只多 6.6% 吞吐、延迟 x1.87（[m5-tts-optimization-log.md:2098](m5-tts-optimization-log.md)）。
- **vocoder 放 CPU**：M5 的 `wall = backbone + max(AR, vocoder)` 结构直接照搬。
- **单帧时序**（目标，v73 配置 E）：

```
t=0    HTP: G2(frame n)  [55 ms]  ├──────────────┤
t=0    CPU: G3(chunk k)  [18 ms]      ├────┤
t=55   CPU: 采样/拼 code、写 ring、唤醒 [2 ms]
t=57   HTP: G2(frame n+1) ...
                                     ^ 80 ms 帧时钟，20 ms 余量
```

### 4.4 抗抖动

- **ring buffer 400 ms**（5 帧），低于 2 帧触发"补水"（插静音/降速）。
- 首帧慢 44% 是 DCVS 升频滞后，不是热漂移（[m5-tts-optimization-log.md:2188](m5-tts-optimization-log.md)）：
  上线前跑 2 次 discard 请求把频点拉起来。
- 并发策略：单 HTP 只保 1 路实时；第 2 路首包会到 3.9 s（M5 实测），产品上要排队限流。

---

## 5. TTFA

### 5.1 分解

```
TTFA = T_speaker + T_prefill + T_frame + T_voc(N=1) + T_mix
```

按 v73 + 配置 E（1-pass predictor, 4 bit talker）+ 优化后 vocoder 算：

| 项 | ms | 瓶颈 | 削减手段 |
|---|---:|---|---|
| T_speaker 命中 / 未命中 | **0 / 120** | MAC（ECAPA 每请求一次） | **按 (voice_id, lang) 缓存**，离线算 |
| T_prefill 10 行（无前缀缓存） | **147** | **算力**（4.4 GMAC / 30 GMAC/s） | 前缀 KV 复用 |
| T_prefill 1 行（前缀命中） | **38** | 带宽（228 MB / 6 GB/s） | （voice, lang）常驻 |
| T_frame1（第 1 帧） | **53** | 带宽 | 同 §2.1 |
| T_voc N=1 | **37** | **算力**（1.66 GMAC / 45 GMAC/s） | 首包 N=1（不能更小） |
| **合计（前缀命中）** | **128** | | **目标值** |
| 合计（前缀未命中） | 237 | | 首请求 / 换音色 |
| 合计（非流式，100 字文本） | 1100+ | 算力（44 GMAC） | 不要用 |

**注意 T_voc 是常数项**：vocoder 的 2.48 GMAC/帧只与音频时长有关，与 chunk 大小无关。
所以"首包用 N=1"只能省掉等待第 2/3/4 帧的时间，省不掉 vocoder 本身的 55 ms。

### 5.2 三个关键点

1. **prefill 走哪条模式，差一个数量级**：
   - **流式（参考默认）**：prompt 只有 9-10 行（role 3 + think/lang/speaker 4 + tts_bos + 首个 text token），**与文本长度无关**。
     1.1 GMAC -> v73 上 25-100 ms；且这 9-10 行对同一 (voice, lang) 完全固定，KV 可预填并跨请求复用，稳态几乎为 0。
   - **非流式（本仓库现状）**：prefill = 8 + n_text 行，与文本长度线性相关。100 字中文 = 108 行 = 44 GMAC，
     v73 上 **1-2 s**，TTFA 直接崩。**所以 HTP 上切流式不是优化项，是可行性项。**
   - 4 bit prefill 只省带宽不省 MAC；**真正有效的是让 prefill 的 GEMM 打满矩阵单元**（v79/HMX 上 10 ms 级）。
2. **第一帧 = 稳态一帧**，没有"首帧更便宜"的路径。所以**稳态 RTF 决定 TTFA 的量级**。
3. **不要等 72 帧**：现在的实现把 TTFA 变成 5.76 s + 一帧，这是 P0 必须修的。

### 5.3 目标

| 场景 | RTF | TTFA | 条件 |
|---|---:|---:|---|
| **v73 全做（目标）** | **0.66** | **128 ms** | P0 + P1 + P2a + P2b |
| v73 只做 LM 侧 | 1.02 | ~210 ms | P0 + P1 + P2a（vocoder 钉住） |
| v73 只做 4 bit | 1.88 | ~320 ms | P0 + P1 |
| v73 现状 | 3.8 | 5.8 s | — |
| 参考：官方 GPU 单流（vLLM + CUDA graph） | 0.288 | 97 ms | arXiv 2601.15621 Table 2 |

最后一行的意义：**这颗模型在数据中心 GPU 上也只有 0.288**（3.5x 实时）。
v73 要 0.66（1.5x 实时），等于要求手机达到 GPU 的 43% —— 这就是为什么必须动模型结构。

---

## 6. 精度

| 项 | 处理 | 依据 |
|---|---|---|
| predictor 的 `ffn_down` | **F32/F16 保底**（已有，[conversion/qwen3tts.py:283](conversion/qwen3tts.py)） | FFN 中间值峰值 1.5e5，超 F16 范围 |
| attention 的 QK^T / AV | 保 fp16（不要 A8W8） | M5 实测量化后 -1.3 dB |
| talker FFN / 投影 | int8 per-channel 可接受，A16W8 更稳 | 主干默认 A16W8 |
| 4 bit | 只对 FFN 的 gate/up/down 与 embed 做，attention 保 8 bit | 6 bit 以下 attention 会改变音色 |
| vocoder 尾层 / codebook 表 | 保 fp32/fp16（已有） | M5 最终形态就是"fp32 只剩 codebook 表" |
| Snake | 9 阶多项式（`sin^2(a) = a^2 - a^4/3 + 2a^6/45`），不用 libm | HVX 无 sin，M5 已验证 -23% |

验收必须三层，缺一层都会翻车：
1. **逐样本**：codes 一致率（>=99.5%）、PCM SNR（>=30 dB）、与 HF 参考逐字节比对（前 3 s）。
2. **客观**：ASR 回译 WER、时长/过零/峰值。
3. **主观**：盲听 20 条，MOS >= 4.0。**量化与换 kernel 必须过耳朵。**

---

## 7. 落地计划

### P0 静态化与流式化（不改模型，1–2 周）

| 改动 | 文件 | 收益 |
|---|---|---|
| **实现流式模式**：prompt 只放固定前缀 + 首字，其余文本按帧喂（`trailing_text_hidden`） | [mtmd-helper-gen.cpp:245-255](tools/mtmd/mtmd-helper-gen.cpp) | 长文本 TTFA 从 O(n_text) 变常数；100 字 1-2 s -> 0.1 s |
| 采样补齐 `temperature=0.9` / `repetition_penalty=1.05` / `min_new_tokens=2` | [qwen3tts-gen.cpp:6](tools/mtmd/models/qwen3tts-gen.cpp) | 与参考对齐，修韵律漂移 |
| `window_frames` 72 -> chunk 调度 1/2/4（左上下文 25-32） | [mtmd-helper-gen.cpp:346,500](tools/mtmd/mtmd-helper-gen.cpp) | TTFA 5.76 s -> 0.5 s |
| G3 按 N 建图（N=1,2,4），不再固定 72 帧算满 | [qwen3tts-gen.cpp:744](tools/mtmd/models/qwen3tts-gen.cpp) | vocoder 计算量 -18x（N=4） |
| c2w state 切 `in_place`，去掉 host 回传 | [qwen3tts-gen.cpp:751](tools/mtmd/models/qwen3tts-gen.cpp), [models.h:535](tools/mtmd/models/models.h) | 每 chunk 省 9.4 MB 搬运 + 1 次同步 |
| talker+pred 融成一张图，采样上移进图 | [mtmd-helper-gen.cpp:320](tools/mtmd/mtmd-helper-gen.cpp) | 每帧省 1 次提交 + 1 次同步（~2–4 ms） |
| talker KV 按 128/256/512 分档，图常驻 | llama-kv-cache / 图缓存 | 避免重建；万帧长文也稳 |
| speaker embedding 按 (voice,lang) 缓存 | [mtmd-helper-gen.cpp:212](tools/mtmd/mtmd-helper-gen.cpp) | TTFA -120 ms（v73） |
| 启动 warmup（跑一句完整句子丢弃） | tts.cpp / server | 首批不再慢 44% |
| `GGML_HEXAGON_OPPOLL=1` | 部署脚本 | 提交路径 -1.8x（M5 实测） |

P0 后：**RTF 不变（字节没变），TTFA 从 5.8 s 到 ~0.5 s（v73）/ ~0.15 s（旗舰）。**

### P1 权重量化（2–3 周）

- talker：FFN 4 bit、attention 8 bit（A16W8）；predictor 先保 8 bit；vocoder 全 int8。
- 目标：LM 150 ms（902 MB/帧）-> wall 150 ms，RTF 1.88。
- **先做一天的门禁实验**：v73 上 q4_0 未必比 q8_0 快。M5 在同一颗芯片实测 640 行 GEMM
  q8_0 31.8 ms vs q4_0 38.3 ms（[m5-tts-port-notes.md:370-383](m5-tts-port-notes.md)），
  因为 dequant 开销大于省下的带宽。**若 q4 不赢，跳过 P1 直接做 P2a + talker 蒸馏**（0.35B W8 = 264 MB）。
- 必须在 P0 之后做，否则测量被调度噪声淹没。

### P2a predictor 并行化（重训，3–4 周 GPU + 1 周部署）

> 训练配方、张量清单、代码改动点、验收门槛见 **[qwen3-tts-predictor-redesign.md](qwen3-tts-predictor-redesign.md)**。

**关键结构洞察**：一旦所有 15 个 head 都只读同一个位置（talker hidden），
predictor 的注意力就退化成"对 1 个元素做 softmax" = 恒等作用于 V。
于是 5 层 transformer **可以整个塌成 5 层纯 FFN 塔**（q/k/softmax 全删），
参数从 81.8 M/step 降到 **89 M（含 15 个 head）总**，一步出 16 个 code，KV 也不需要了。

| 方案 | 做法 | 每帧字节 | 风险 |
|---|---|---:|---|
| A | 16 个并行 lm_head，一次前向 | 92 MB | 中（分布偏移） |
| B | 分组 2–3 次前向，每次 5–8 个 code | 180–270 MB | 低（推荐先做） |
| C | 蒸馏到 3 层 x 512 宽，保留 AR | 337 MB | 低，但收益只有 1/4 |

- 训练信号：用**原模型在语音语料上产出的 (talker hidden, 16 codes) 对**做教师强制蒸馏，
  不需要文本、不需要 TTS 标注；5 层网络的训练成本很低（M5 的 S0 判据是 8 GPU-小时）。
- 立项判据：codes top-1 一致率 >= 60%，且盲听无退化。
- 预期：LM 53 ms（4 bit）/ 92 ms（8 bit），每帧字节 1348 -> 92 MB，算子 800 -> ~150。

### P2b vocoder 减重（重训，3–4 周）

94% 的 MAC 在 12 个残差单元上，这是唯一值得动的杠杆：

| 手段 | 预期 | 风险 |
|---|---:|---|
| res unit 3 -> 2（每 block） | -25% MAC | 低，音质略降 |
| 通道 768/384/192/96 -> 640/320/160/80 | -13% MAC | 低 |
| 全 int8 + Snake 多项式/LUT | -30~40% 时间 | 低（M5 已验证） |
| CT -> MatMul 恒等分解 | -25% kernel 时间 | 低（SNR 30 dB，M5 验证） |

- 目标：82 ms -> **55 ms**（2.48 -> 1.6 GMAC/帧）。
- **不做 P2b，wall 会被钉在 82 ms，RTF 停在 1.02** —— 这是本项目最容易踩的坑。

**P2a + P2b 后：wall 53 ms（LM 受限），RTF 0.66，TTFA 128 ms。**

### P3 QNN context binary（量产）

- 一张 context、6–8 个 graph、共享权重（v73+、单 PD、<=64 graph）。
- 开 DLBC（带宽压缩）与 Sparse Weights，直接降 DDR 流量。
- 换 HMX 机型后，把 attention/FFN 全部走 fp16 HMX（12 TFLOPS），G2 从"带宽受限"变"算力富余"。

---

## 8. 测量协议

1. **先量斜率，再谈优化**：`每帧字节 -> ms` 做两点拟合（N=1 与 N=4 各一组），确认是带宽受限还是调度受限。
2. **三段拆分**：`enqueue / wait / kernel` 分开记（M5 的 734-op 图 enqueue 2.6 ms、step 100 ms，只看 host 时间会得出错误结论）。
3. **必须 warmup 3 次 discard**，交替 A/B 正逆序，取中位数。
4. **RTF 只在输出长度不变时可比**；M5 出现过"墙钟 +5.1% 而 RTF -4.9%"的反例。
5. 先测 **DDR 带宽探针**（大块 1D/2D 拷贝 + 640 行 q8 GEMM），把本文表格里的 4.5/8/16/30 GB/s 换成真机数字，场景表立即变成真机预测。

---

## 9. 开放问题

**已定**：目标机型 v73（SM6650）；允许重训。因此 P2a、P2b 都在范围内。

剩下的是工程门禁，按"做错了会白干一周"的顺序排：

1. **实测 v73 的有效带宽和 vocoder 的 int8 吞吐**（半天）。这两个数直接决定 RTF 是 0.66 还是 0.89：
   - 带宽 < 5 GB/s -> LM 侧 4 bit 都不够，必须叠 talker 蒸馏。
   - vocoder int8 < 35 GMAC/s -> P2b 的减重目标要从 -33% 提到 -45%。
2. **q4 在 v73 上到底赢不赢**（1 天）。M5 的探针显示 q4_0 反而慢于 q8_0；若复现，P1 直接跳过，改走 talker 蒸馏 0.35B。
3. **QNN SDK 有没有**。没有的话 P3 顺延，P0–P2 用 ggml-hexagon 的固定 shape 图跑，但那时"2050 算子 -> 760 算子"的合并是硬要求。
4. **vocoder 允许多大音质损失**。res unit 3->2 是 -25% MAC 的最大单点，但也是唯一可能被耳朵否决的一项；需要先做 20 条盲听的 A/B 基线。
5. **16 个 codebook 是否都要**（降级档）：只喂前 8 个省 50% predictor 字节，但音色会闷；作为"最便宜的降级开关"保留。
6. **对话式场景的文本是流式到达的**：可以先预填首句、边生成边等下一句，TTFA 只按首句算；这要求 prompt 支持增量 prefill（当前实现是整段一次性 prefill）。

---

## 附 A：这些数字怎么复现

[qwen3-tts-roofline.py](qwen3-tts-roofline.py) 就是本文 §2.1 那张表的生成脚本：

```sh
python3 docs/backend/snapdragon/qwen3-tts-roofline.py
```

改脚本里 `cfg` 的元组 `(talker 位宽, predictor 位宽, pass 数, vocoder 位宽, chunk, KV MB)`
就能算自己的方案。真机实测带宽出来以后，把 `(4.5, 8, 16, 30)` 换成实测值即可。

## 附 B：参考

| 内容 | 出处 |
|---|---|
| 模型结构 / 配置 | Qwen3-TTS-12Hz-0.6B-Base 的 `config.json`、`speech_tokenizer/config.json` |
| 参考实现（生成循环、chunked_decode、采样默认值） | https://github.com/QwenLM/Qwen3-TTS ，`qwen_tts/core/models/modeling_qwen3_tts.py`、`qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` |
| HTP 静态图 / variable batch / weight sharing / graph switching / DLBC | https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html |
| QNN EP 的静态 shape 限制与 QDQ 精度档位 | https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html |
| HMX 32x32 tile、batch=1 利用率、v75 峰值 | https://arxiv.org/html/2509.23324v1 |
| 本仓库真机数据（v73、每算子 137 µs、Dual-Track 之外的 M5 经验） | [m5-tts-optimization-log.md](m5-tts-optimization-log.md)、[m5-tts-port-notes.md](m5-tts-port-notes.md)、[m5-vocoder-npu-eval.md](m5-vocoder-npu-eval.md) |

