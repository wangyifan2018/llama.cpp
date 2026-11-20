# code_predictor 并行化改造规格（P2a）

目标机型已定 **v73（SM6650）**，允许重训。本文是 [qwen3-tts-htp-streaming-design.md](qwen3-tts-htp-streaming-design.md) §7 P2a 的可执行版本：
把每帧 16 次串行前向压成 1 次，索引从"73% 的字节、80% 的算子数"降到个位数。

---

## 1. 为什么必须改

现状（[qwen3tts-gen.cpp:195-289](tools/mtmd/models/qwen3tts-gen.cpp)）每帧：

| 项 | 值 |
|---|---:|
| 前向次数 | 16（位置 0 填 KV、位置 1 出 code1、…、位置 15 出 code15） |
| 每次读权重 | 81.8 MB（5 层 x 15.73 M + 1 个 lm_head） |
| 每帧读权重 | **1348 MB**（占全模型 73%） |
| 每帧算子数 | **~800**（占 80%） |
| 每帧 MAC | 1310 M |
| KV | 5 层 x 2 x 1024 x 16 行，每帧从零重建 |

根因：15 个 code 严格自回归，第 g 个依赖第 g-1 个**采样结果**，所以只能一次算一行。
每一行的算术强度是 1 MAC/byte，而 HTP 需要 10–30 才不浪费。

## 2. 目标结构：5 层 FFN 塔 + 15 个并行头

### 2.1 一个可证的简化

如果把 15 个 head 都改成只读**同一个位置**（talker 投影后的 hidden），
那 predictor 的注意力就变成"对长度为 1 的序列做 softmax"：

```
attn(x) = W_o · (softmax([q·k/√d]) · v) = W_o · (1.0 · v) = W_o · W_v · x
```

q、k、softmax 全是恒等/常数，**可以整段删掉**。5 层 transformer 塌成 5 层残差 FFN，
参数从 81.8 M/位置 降到 11.54 M/层：

| 张量 | 形状 | 参数 |
|---|---|---:|
| `v_proj`（原 k/v 只留 v） | 1024 x 1024 | 1.05 M |
| `o_proj`（2048 -> 1024 变 1024 -> 1024） | 1024 x 1024 | 1.05 M |
| `gate/up/down` | 1024x3072 x3 | 9.44 M |
| **每层** | | **11.54 M** |
| **5 层** | | **57.7 M** |
| 15 个 `lm_head`（已存在，`A_GEN_CODE_HEAD`） | 1024 x 2048 x 15 | 31.5 M |
| **合计 / 帧** | | **89.2 M = 92 MB** |

对比：**1348 MB -> 92 MB（-93%），800 算子 -> ~150，KV 直接消失。**

### 2.2 图结构

```
h[1024] (talker hidden, 已 project_in)
  └─ rmsnorm ─ v_proj ─ +res ─ rmsnorm ─ gate/up ─ swiglu ─ down ─ +res     x 5 层
       └─ rmsnorm(mtp_norm)
            └─ mul_mat(head_w[1024,2048,15], h[1024,15])   <- 一次 batch matmul
                 └─ 15 x (top_k / top_p / softmax / cumsum / draw)  <- 形状全同，可批
                      └─ 15 个 code + code0 -> 16 x get_rows + 求和 -> 下一帧 talker 输入
```

三点说明：

1. **`A_GEN_CODE_HEAD` 的 3D 布局可以直接复用**（`[n_embd, vocab, 15]`），
   `ggml_mul_mat` 对 ne[2]=15 的权重本来就是按 batch 做的，一次出 15 组 logits。
2. **采样留在图内**：当前实现已经在图内做 top-k/top-p（[qwen3tts-gen.cpp:6-68](tools/mtmd/models/qwen3tts-gen.cpp)），
   保留它就能维持"每帧 1 次提交、0 次同步"。代价是 15 份 ~5 op = 75 个采样算子；
   若嫌多，可改成输出 logits[15,2048]（123 KB）回 host 采样，换一次 ~0.3 ms 的同步。
3. **KV 与 `A_GEN_CODE_ATTN_*` 全部作废**，`list_c2w_state_slots` 与 predictor 的 state 槽一并删除，
   静态图里少 20 个 state 张量。

## 3. 三个方案怎么选

| 方案 | 结构 | 每帧字节 | 质量风险 | 训练难度 | 建议 |
|---|---|---:|---|---|---|
| **A 全并行** | 1 次 FFN 塔 -> 15 头 | **92 MB** | 中（15 个 code 的组合一致性） | 中 | 目标形态 |
| **B 分组** | 第 1 次出 code1-8，第 2 次（条件=前 8 个 emb 之和）出 9-15 | 180 MB | 低 | 低 | **先做这个** |
| C 蒸馏 | 3 层 x 512，保留 AR 15 步 | 337 MB | 低 | 低 | 只作兜底 |

理由：RVQ 的 16 个 code 是**粗到细**的，靠后的 code 强依赖靠前的 code。
A 方案让 head g 只看 talker hidden，等于假设"15 个 code 在给定语义后条件独立"，
这个假设在 RVQ 上不成立但可能"够用"；B 方案把依赖保留成 2 段，风险小一半、成本只多一倍。

**判据（先跑 S0 再决定）**：用原 AR predictor 当教师，在 2000 帧上算 top-1 一致率。
A 方案若 < 50%，直接转 B。

## 4. 训练配方

### 4.1 数据：不需要文本，不需要 TTS 标注

只要 **(talker hidden, 16 codes)** 对。用原模型在纯语音语料上跑一遍即可：

```
for wav in corpus:                       # 1000-5000 h 中文，16 kHz 也行（重采样到 24k）
    codes = tokenizer.encode(wav)        # [T, 16]
    hidden = talker.forward(prompt, codes)  # 教师强制，取每帧最后位置的 hidden
    save (hidden[t], codes[t])           # 1 帧 = 1 个样本
```

- 每帧一个样本，seq_len=1，训练极便宜：89 M 参数、单帧前向，**8 张卡几小时**。
- 100 h 语音 ≈ 450 万帧，足够（M5 的同类改造用了 >=100 h 商用语料）。

### 4.2 损失

```
L = Σ_{g=1..15} w_g · CE(logits_g, code_g^teacher)
w_g = 1.0 for g<=4, 0.5 for g>=5      # 前几个 code 决定音色/内容，权重高
```

可选加一项"嵌入一致性"：`||Σ_g emb_g(pred) - Σ_g emb_g(teacher)||^2`，直接约束回灌给 talker 的那个和向量。

### 4.3 初始化与冻结

- **热启动**：第 0 位置的 v_proj/o_proj/mlp 直接用原 predictor 第 0 层的权重（形状完全一致，只删 q/k）；
  15 个 head 用原来的 15 个 `lm_head`。
- **冻结**：talker、codec、embedding 全部冻结，只训塔 + head。
- 学习率 1e-4，cosine，batch 512 帧，20 epoch。

## 5. 验收门槛

| 层 | 指标 | 门槛 |
|---|---|---|
| 逐码 | top-1 一致率（vs 教师） | >= 60% |
| 逐码 | 每个 codebook 的熵 / 重复率 | 不低于教师的 80%（防坍缩到高频码） |
| 音频 | PCM SNR（vs 原模型同文本） | >= 25 dB |
| 主观 | 20 条盲听 A/B，MOS 差 | < 0.2 |
| 端到端 | wall | <= 55 ms（v73） |

**只过前两项不许上线**：codes 一致率高但音质塌的例子在 RVQ 上很常见（靠后的 book 全变 0 也能有 60% 一致率）。

## 6. 代码改动清单

| 位置 | 改动 |
|---|---|
| [conversion/qwen3tts.py](conversion/qwen3tts.py) | 写 GGUF 时打一个 `qwen3tts.mtp_mode = "parallel"` KV；沿用 `A_GEN_CODE_HEAD`，不再写 `A_GEN_CODE_ATTN_*`；`codec_embedding` 仍需 15 份 |
| [qwen3tts-gen.cpp:195-289](tools/mtmd/models/qwen3tts-gen.cpp) | `prefill()`/`step()` 换成 `mtp_parallel()`；删 k/v cache 参数；15 头一次 matmul；采样循环改 15 次同形状 |
| [qwen3tts-gen.cpp:616-662](tools/mtmd/models/qwen3tts-gen.cpp) | `list_c2w_state_slots` 里去掉 predictor 的 state（本来也没有），确认 talker 的 KV 仍由 llama 管 |
| [mtmd-model.cpp:1073](tools/mtmd/mtmd-model.cpp) | `PROJECTOR_TYPE_QWEN3TTS_GEN` 分支按 `mtp_mode` 选 hparams |
| [mtmd-helper-gen.cpp:333](tools/mtmd/mtmd-helper-gen.cpp) | GEN_CODE 调用点不变（输入 h、输出 codes+embd），只是内部实现变 |

**兼容性**：按 KV 里的 `mtp_mode` 分派，老 GGUF（AR 模式）继续可跑，方便 A/B。

## 7. 风险与回退

| 风险 | 表现 | 回退 |
|---|---|---|
| A 方案音质塌 | 高频 book 全坍缩、发闷 | 转 B 方案（2 次前向，180 MB，仍 -87%） |
| codes 一致率低 | < 50% | 加"嵌入一致性"损失；或转 C（3 层 512 宽，337 MB） |
| talker hidden 分布漂移 | 换音色/语言后一致性骤降 | 数据集必须覆盖全部 10 种语言与目标音色 |
| 训练不收敛 | 5 层塔容量不够 | 加到 6-7 层（每层 11.5 M，仍只有 80 MB） |

**最坏情况**：回退到 C（337 MB），LM 侧 92 -> 337 MB，wall 61 -> 92 ms，
再叠 talker 蒸馏（0.35B）仍可回到 0.8 附近。**这个方案不会把项目做死。**
