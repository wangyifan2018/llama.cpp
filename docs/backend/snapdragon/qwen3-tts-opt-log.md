# Qwen3-TTS 0.6B 优化实验记录（v73 / SM6650）

> 目标：RTF < 1，TTFA 尽量短。每次优化一行数据、一段结论、一条复现命令。
> 设计背景见 [qwen3-tts-htp-streaming-design.md](qwen3-tts-htp-streaming-design.md)，P2a 规格见 [qwen3-tts-predictor-redesign.md](qwen3-tts-predictor-redesign.md)。

平台说明：

| 平台 | 用途 | 后端 | 备注 |
|---|---|---|---|
| MacBook M2 8 GB | 结构/等价性/相对耗时 | **纯 CPU**（`build-host-cpu`） | Metal 在这台机器上不可用，见 E00-b |
| SM6650（v73） | 目标机 | arm64 CPU（`build-android-tts-cpu`） | HTP 需要 hexagon SDK，本机未装 |

约定：
- **RTF = wall / 音频时长**；**TTFA = 首个真实 PCM 样本 - 请求开始**（不含程序启动）。
- 所有 A/B **必须固定 `--seed`**，否则两次跑的是两句话（E03 踩过）。
- 每轮固定 `-p "Hello, this is a test." --tts-lang english`、`-t 6`（host）/ `-t 4`（device）。

---

## 0. 测量工具

给 `llama-tts` 加了三处**只读**instrumentation（默认关闭，不影响正常使用）：

| 开关 | 作用 | 代码 |
|---|---|---|
| `TTS_STREAM=1` | 逐帧 `get_output()`，打印 TTFA 与 RTF | [tts.cpp](../../../tools/tts/tts.cpp) |
| `TTS_CHUNK=N` | code2wav 每次解码 N 帧（默认 72 = 整个注意力窗） | [mtmd-helper-gen.cpp](../../../tools/mtmd/mtmd-helper-gen.cpp) |
| `TTS_TRACE=1` | 打印每帧 `talker decode` 与 `gen_code(predictor)` 的 ms | [mtmd-helper-gen.cpp](../../../tools/mtmd/mtmd-helper-gen.cpp) |

复现一条基线：

```sh
./build-host-cpu/bin/llama-tts \
  -m /tmp/q3tts-models/qwen3-tts-0.6b-q8.gguf \
  -mm /tmp/q3tts-models/mmproj-qwen3-tts-0.6b-f16.gguf \
  -p "Hello, this is a test." --tts-lang english \
  --tts-speaker-file <ref.wav> --seed 1234 -n 24 -t 6 -o out.wav
```

---

## 1. 实验索引

| # | 假设 | 改动 | 结果 | 状态 |
|---|---|---|---|---|
| E00 | 环境可用性 | 转换 GGUF、建 CPU 后端 | Metal 比 CPU 慢 **200x**，改用 CPU | 结论 |
| E01 | 基线 | 不改 | RTF 3.97，vocoder 占 67% | 完成 |
| E02 | 每帧都在算 72 帧的 vocoder | graph 按实际帧数建 | vocoder 5.09 → **1.41 s**，bit-exact | 完成 |
| E03 | chunk 越小 TTFA 越好 | `TTS_CHUNK=2..72` 扫描 | TTFA 4.29 → 1.44 s，wall +17% | 完成 |
| E04 | 每帧串行链的成本分布 | `TTS_TRACE` | predictor 64 ms、talker ~60 ms | 完成 |
| E05 | chunk=1 更省？ | `TTS_CHUNK=1` | 反例：+100 ms/帧 | 完成 |
| E06 | 分块会改变音质？ | 固定 seed 比对 | **SNR 215 dB，逐位相同** | 完成 |
| E07/E08 | 设备能测吗？ | 上机、加预热、绑核 | 同命令 **35–50x** 抖动 | 见 E09/E10 |
| E09 | 热态基线是多少 | 冷态作废、取快模态 | RTF **5.4**（不是之前说的 49） | 完成 |
| E10 | 抖动是不是页缓存/绑核/加载模式 | 预加载、taskset、load-mode | 全排除；慢在 **mtmd 生成图内部** | 待插桩 |

---

## E00 准备（一次性）

### E00-a 模型转换

```sh
curl -L "https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base/resolve/master/{model.safetensors,speech_tokenizer/model.safetensors}"
PYTHONPATH=/tmp/pylibs python3 convert_hf_to_gguf.py /tmp/q3tts-models --outtype q8_0 --outfile qwen3-tts-0.6b-q8.gguf
PYTHONPATH=/tmp/pylibs python3 convert_hf_to_gguf.py /tmp/q3tts-models --mmproj --outtype f16 --outfile mmproj-qwen3-tts-0.6b-f16.gguf
```

产出：backbone 646 MB（Q8_0）+ mmproj 621 MB（F16）。hf-mirror 有 429 限流，modelscope 可直连。

### E00-b 后端陷阱：这台 M2 上 Metal 比 CPU 慢 200 倍

| 后端 | pp16 | tg8 |
|---|---:|---:|
| MTL,BLAS（默认构建） | 6.97 t/s | **0.42 t/s** |
| CPU（`-DGGML_METAL=OFF`） | 683.8 t/s | **88.3 t/s** |

**结论**：本机 GPU 路径不可用（内存压力/Metal 驱动，8 GB 机器 swap 已用 5.7 GB），
所有 host 测量必须用 `build-host-cpu`。这不是模型问题，不要在这上面浪费时间。
**复现**：`llama-bench -m qwen3-tts-0.6b-q8.gguf -p 16 -n 8 -t 6`。

---

## E01 基线：RTF 3.97，vocoder 占三分之二

```sh
TTS_TRACE=1 ./llama-tts ... -n 24 -t 6      # 原始代码
```

| 阶段 | 时间 | 占比 |
|---|---:|---:|
| prompt eval（含 speaker encode） | 0.25 s | 3% |
| generation（LM：talker + predictor ×24 帧） | 2.28 s | 30% |
| **vocoder（一次 72 帧图）** | **5.09 s** | **67%** |
| 合计 / 音频 1.92 s | 7.62 s | RTF **3.97** |

两个观察：

1. **vocoder 是 wall 的大头**，与设计文档 §2.3 的预测一致（2.48 GMAC/帧是算力题）。
2. 24 帧的输出调了一次 **72 帧**的图 —— 有 2/3 的算力是纯浪费。

---

## E02 去掉尾部浪费：graph 按实际帧数建，vocoder 5.09 → 1.41 s

**假设**：`build()` 里 `n_frames = hparams.wav_tfm_swa`（固定 72）导致短句按 72 帧算。
**改动**（4 处）：

| 文件 | 改动 |
|---|---|
| [models.h](../../../tools/mtmd/models/models.h) | `mtmd_graph_qwen3tts_gen` 增加 `n_frames` |
| [mtmd-graph.cpp](../../../tools/mtmd/mtmd-graph.cpp) | 传 `params->codes->size()/n_cb` |
| [qwen3tts-gen.cpp](../../../tools/mtmd/models/qwen3tts-gen.cpp) | `build()` 用 `this->n_frames` |
| [mtmd-subcontext.cpp](../../../tools/mtmd/mtmd-subcontext.cpp) | 不再右补 code0；去掉按 72 帧裁尾 |

**结果**（24 帧输出，固定 seed）：

| | vocoder | total | RTF |
|---|---:|---:|---:|
| 改前（72 帧图） | 5.09 s | 7.62 s | 3.97 |
| 改后（24 帧图） | **1.41 s** | 4.15 s | **2.16** |

**音频完全一致**（见 E06），所以这是**零代价的 -46% wall**。
**教训**：固定 shape 不等于"固定成上界"；上界只在首包有用，稳态必须贴合实际长度。

---

## E03 chunk 扫描：TTFA 与 RTF 的取舍曲线

24 帧输出，`TTS_STREAM=1`，逐档扫 `TTS_CHUNK`：

| chunk | prompt | 生成 | voc(outside) | total | **TTFA** | RTF |
|---:|---:|---:|---:|---:|---:|---:|
| 72（一次） | 0.21 | 2.42 | 1.66 | 4.29 | **4.29** | **2.23** |
| 16 | 0.21 | 3.67 | 0.59 | 4.48 | 2.87 | 2.33 |
| 8 | 0.32 | 4.63 | 0.00 | 4.95 | 2.10 | 2.58 |
| **4** | 0.23 | 4.80 | 0.00 | 5.03 | **1.44** | 2.62 |
| 2 | 0.21 | 6.74 | 0.00 | 6.96 | 1.26 | 3.62 |

读法：

- **TTFA 随 chunk 单调下降**（4.29 → 1.44 s，-66%），代价是 wall +17%（chunk=4）。
- **chunk ≤ 2 开始崩**（chunk=2 的 wall +39%），chunk=1 更差（E05）。
- **chunk=4 是拐点**：TTFA 1.44 s，RTF 2.62。若要再压 TTFA，得先削每帧的 LM 成本，而不是继续切小。

**复现**：`for C in 72 16 8 4 2; do TTS_STREAM=1 TTS_CHUNK=$C ./llama-tts ... -n 24; done`

---

## E04 每帧串行链的成本分布（`TTS_TRACE=1`）

稳态每帧（12 帧样本，chunk=4）：

| 段 | ms/帧 | 说明 |
|---|---:|---|
| **gen_code（predictor，16 次前向）** | **64** | 与设计文档"73% 字节、80% 算子"吻合 |
| talker decode + 采样 + 回灌 | ~60 | 比 llama-bench 的 11.3 ms/token 高 5 倍，待查（见 §3 下一步） |
| 合计 LM | ~124 | |
| vocoder（摊到每帧，chunk=4） | ~60-100 | 与 2.48 GMAC/帧 ÷ ~35 GMAC/s 一致 |

首帧例外：`gen_code` 前 3 帧是 144 / 94 / 72 ms，之后稳定在 64 ms —— **图构建与首次执行的固定成本 ~80 ms**，这就是 warmup 的价值。

**结论**：predictor 是 LM 里最大的一块（52%），干掉它（P2a）能把 LM 从 124 → 60 ms/帧。

---

## E05 反例：chunk=1 不是"最流式"，是最慢

8 帧输出，等工作量，只改调用次数：

| chunk | 调用次数 | total | 相对 |
|---:|---:|---:|---:|
| 8 | 1 | 1.62 s | 1.00x |
| 4 | 2 | 1.50 s | 0.93x |
| 2 | 4 | 1.68 s | 1.04x |
| 1 | 8 | **2.56 s** | **1.58x** |

chunk=1 时 vocoder 的 T=1，DAC 残差单元（k=7 卷积）完全没有向量化，**+100 ms/帧**。
**结论**：流式粒度不能小于 4 帧；TTFA 的进一步优化要靠削 LM，不是削 chunk。

---

## E06 验证：分块解码是逐位相同的

最初比对得到 **-4.7 dB**，看起来像"分块改变了音质"。
实际原因：`--seed` 默认随机（`UINT32_MAX`），两次跑的是两段不同的语音。

固定 `--seed 1234` 后：

```
chunk72 vs chunk8: SNR = 215.0 dB, max|diff| = 0.0000
chunk72 vs chunk4: SNR = 215.0 dB, max|diff| = 0.0000
```

**结论**：

1. code2wav 的分块解码与整段解码**逐位等价**（因为全因果 + 71 帧左状态），设计文档的前提成立。
2. **教训**：TTS 的 A/B 必须固定 seed，否则噪声项就是整句话。

---

## 2. 到目前为止的净收益（host CPU，M2 6 线程）

| 版本 | wall | RTF | TTFA |
|---|---:|---:|---:|
| 原始 | 7.62 s | 3.97 | 5.76 s（等 72 帧） |
| + 精确帧数图（E02） | 4.15 s | 2.16 | 4.29 s |
| + chunk=4（E03） | 5.03 s | 2.62 | **1.44 s** |
| + 训练后 predictor（预估，P2a） | ~3.0 s | ~1.6 | ~0.9 s |

（wall 在 chunk=4 时略升，因为每帧的 vocoder 固定开销；这是用 wall 换 TTFA，值得。）

## 3. 下一步（按信息量排序）

1. **talker decode 为什么是 60 ms 而不是 11 ms**（llama-bench 同模型 tg=88 t/s）。
   怀疑：每帧重建 decode 图 / 155008 词表的 lm_head 全量计算 / 采样器开销。
   先测：给 `llama_process` 内部加计时，或换 `-n 1` 单帧对比。
2. **vocoder 每调用固定开销**（chunk=4 时每帧多 ~40-90 ms）：graph 是否每次重建？
   若是，加按 (N, shape) 的图缓存 + state 改 `in_place`。
3. **上设备**：arm64 CPU 镜像已构建，模型与二进制推送到 `/data/local/tmp/q3tts`，跑同一组命令，得到 v73 的真实 CPU 基线。
4. **P2a predictor 并行头**：E04 已量化其价值（64 ms/帧 → 预计 ~5 ms），是最值得做的重训。

---

## E07 设备（v73）基线：冷热差 5–50 倍，A/B 必须先预热

**注意**：设备二进制构建于 01:38，**早于 E02/E03**，所以这一节是"原始代码"的数字。

同一份命令（`-n 8 -t 4`），连续两次跑，中间只隔一次 push：

| 轮次 | prompt eval | generation（8 帧） | vocoder | total |
|---|---:|---:|---:|---:|
| 第 1 轮（F16 mmproj） | 0.09 s | **1.87 s**（234 ms/帧） | 22.20 s | 24.2 s |
| 第 2 轮（Q8 mmproj） | **12.25 s** | 68.3 s（8.5 s/帧） | 21.03 s | 101.6 s |

prompt eval 从 0.09 s 变成 12.25 s，**说明两轮的机器状态完全不同**（DCVS 降频 / 热节流 / 页缓存），
所以 Q8-vs-F16 的对比**不成立**。这与 M5 记的"静置后首批请求慢 44%"是同一现象，只是这里更极端。

**方法论结论**：设备上做 A/B 必须
1. 先跑 2–3 次 discard 把频点拉起来；
2. 交替 A/B/A/B（不是 A/B 各一次）；
3. 取中位数，并同时记录 prompt eval 作为"机器状态指纹"——它若差 10 倍，这一轮就作废。

### 唯一可靠的设备数字（热态）

| 段 | 8 帧 | 每帧 | 说明 |
|---|---:|---:|---|
| generation（LM） | 1.87 s | 234 ms | 其中 predictor 占大头 |
| **vocoder（72 帧图）** | **22.20 s** | **308 ms** | **RTF 3.85，单独就超实时 4 倍** |

**vocoder 是设备上的绝对瓶颈**：一次调用固定算 72 帧（原始代码），8 帧输出也要付 22 秒。
E02 的"按实际帧数建图"正是治这个 —— 8 帧的图应该只要 ~2.5 s（22.2 × 8/72），
但**必须在设备上重测确认**，因为小 T 的效率可能更差（E05 在 host 上已看到）。

---

## 4. 结论汇总（截至本轮）

1. **host CPU 上已拿到两个可靠的 P0 收益**：
   - 精确帧数图：vocoder 5.09 → 1.41 s（-72%），wall -46%，**bit-exact**；
   - chunk=4：TTFA 4.29 → 1.44 s（-66%），wall +17%。
2. **设备上第一个要修的不是模型，是"每次调用固定算 72 帧"**：8 帧输出付 22 秒 vocoder。
3. **设备 A/B 的噪声（冷热差 5–50 倍）比要测的效应还大**，必须建立预热 + 交替 + 指纹校验的流程，
   否则会得出"Q8 比 F16 慢 4 倍"这种假结论（本轮就差点得出）。
4. mtmd 自定义图在 A78 上的算子效率仍未定量（F16/Q8 对比被噪声毁掉），
   需要用 `GGML_HEXAGON_PROFILE` 或 op 级计时单独测，而不是靠端到端 wall。

## 5. 下一步（按优先级）

| # | 实验 | 为什么先做 |
|---|---|---|
| 1 | **设备预热协议**（3 次 discard + 交替 + prompt-eval 指纹） | 不做这个，后面所有结论都不可信 |
| 2 | 设备上重测 E02（精确帧数图） | 预期 vocoder 22.2 → ~2.5 s，是最大单点 |
| 3 | 设备上重测 chunk=4 | TTFA 从 22 s 降到 ~3 s |
| 4 | op 级剖析：`ggml_conv_1d` / `mul_mat(F16)` / `top_k` 在 A78 上的实际吞吐 | 决定要不要换量化类型或换 kernel |
| 5 | 修掉 host 上 talker decode 的 5 倍异常（60 ms vs llama-bench 11 ms） | 影响所有平台的 40% LM 时间 |

---

## E08 设备测量本身是最大的障碍：同一命令 50 倍抖动

改完 E02 后重建 arm64 二进制并推到设备，用"预热 + 交替 A/B"重测。结果先看到的是噪声。

同一份命令（8 帧、chunk=72、`-t 4`、`--seed 1234`），在一小时内跑出的分布：

| 指标 | 最快 | 最慢 | 倍数 |
|---|---:|---:|---:|
| prompt eval | 0.08 s | 13.14 s | **164x** |
| generation（8 帧） | 1.85 s（231 ms/帧） | 91.75 s（11.5 s/帧） | **50x** |
| vocoder | 1.64 s | 14.35 s | 8.7x |

排查过、**都不是原因**：

| 假设 | 实验 | 结果 |
|---|---|---|
| 线程同步开销 | `-t 1 / 2 / 4` | 单线程更慢（8.5 s/帧），排除 |
| 大核/小核调度 | `taskset f0`（绑 4-7 大核）vs 不绑 | 绑了反而更慢（76.8 s），排除 |
| DVFS 降频 | 0.5 s 采样 `scaling_cur_freq` | 跑的时候大核稳定在 2304 MHz，排除 |
| 内存不足 | `MemAvailable` | 始终 8 GB 以上，排除 |
| 加载模式 | `--load-mode auto/none/mlock` | 三种都有快有慢，不相关 |

**唯一稳定的相关性：慢轮次的 `prompt eval` 一定是 12–13 s。**
15 个 token 的 prefill 不可能要 12 s —— 说明那 12 s 花在**把 646 MB 权重从存储读进内存**（约 50 MB/s，典型的 4 KB 缺页）。
换句话说：**模型页被回收了，进程在运行时持续缺页**。

### 结论与协议

1. **v73 上的绝对性能数据，在页缓存命中时才有意义**。判断方法：**看 `prompt eval`**。
   - `prompt eval` ≈ 0.1 s -> 热态，数据可用；
   - `prompt eval` ≈ 12 s -> 冷态（权重在存储里），**该轮作废**。
2. 冷态的 50 倍惩罚不是模型问题，是**部署问题**：mmap 的 GGUF 在 adb shell 进程里得不到页缓存保护。
   量产必须：`mlock` / `--load-mode none` / 启动时预热一遍全部权重，三者至少一个，否则首请求必然灾难。
3. 之前记的"RTF 49"是**全冷态**数字，应当作废。热态基线见下。

## E09 热态基线（v73，8 帧，chunk=72，含 E02 修复）

| 段 | 时间 | 每帧 | 目标（80 ms/帧） | 差距 |
|---|---:|---:|---:|---:|
| generation（talker + predictor） | 1.85 s | **231 ms** | 80 ms | 2.9x |
| vocoder（8 帧图） | 1.64 s | **205 ms** | 80 ms | 2.6x |
| 合计 | 3.57 s（音频 0.64 s） | 436 ms | 80 ms | **RTF 5.4** |

E02 的效果在设备上得到确认：**vocoder 从"一次 72 帧图 22.2 s"降到"8 帧图 1.64 s"**（-93%）。

对照 host（M2 6 线程，同样的 chunk=72、12 帧）：talker 11 ms + predictor 65 ms + vocoder 60 ms/帧 = 136 ms/帧。
**设备比 host 慢 3.2 倍**，这个比例是合理的（A78 vs M2），说明**热态下没有"神秘 100 倍"**，
之前的异常全部来自冷态。

### 修正 E04 的一个错误

E04 曾写"talker decode ~60 ms，比 llama-bench 高 5 倍，待查"。加了 trace 后实测：

| 段 | host 实测 |
|---|---:|
| talker decode | **11 ms**（与 llama-bench 88 t/s 完全一致） |
| gen_code（predictor） | **65 ms** |
| 差额（~50 ms） | chunk=8 时有一次 vocoder flush 落在 generation 里 |

**没有异常**，是之前用减法推断错了。LM 每帧 76 ms 里 predictor 占 85%。

## 6. 本轮之后的判断

1. **v73 的 CPU 路线：RTF 5.4 起步**，LM 和 vocoder 各差 2.6–2.9 倍。
   即使 P2a（predictor 并行，65 → ~8 ms）和 P2b（vocoder -33%）都做完，
   CPU 上最好也只能到 **RTF ~2.5** —— 与设计文档的结论一致：**v73 必须用 HTP，CPU 只能做兜底**。
2. **设备测量的门槛已经迈过去了**（用 prompt eval 做指纹 + 冷态作废），下一轮的数据才是可比的。
3. 下一轮优先级改为：
   - **P0-a**：给 v73 的部署加"预热全部权重"（或 mlock），把冷态惩罚从 50x 抹平 —— 这是产品可用性的前提，比任何模型优化都优先；
   - **P0-b**：设备上确认 chunk=4 的 TTFA（热态），把 TTFA 从"等一整个窗"降到 ~4 帧；
   - 然后才回到 P2a 重训。


---

## E10 续查：预加载、绑核、加载模式都排除了，抖动是"每进程"的

同一命令（8 帧、chunk=72、`-t 4`、同 seed）连续跑 12 次，`generation` 的分布：

```
1.85  1.86  1.87  2.36  4.01  13.5  60.2  61.4  68.4  68.9  76.8  91.8   (秒)
```

**两个模态差 35–50 倍**，而且会在相邻两次运行之间翻转（同一脚本内 preload2 慢 61 s，preload3 只需 1.86 s）。

### 又排除的三个因素

| 假设 | 实验 | 结果 |
|---|---|---|
| 权重不在页缓存 | 每次运行前用 `cat` 把两个 gguf 顺序读一遍 | `prompt eval` 从 12.2 s 降到 0.12 s（backbone 确实被预热了），但 `generation` 仍 68.9 s |
| 大核调度 | `taskset f0`（4-7 大核） | 更慢（76.8 s） |
| 加载模式 | `--load-mode auto / none / mlock` | 三种都有快有慢 |

**关键观察**：preload1 的 `prompt eval` 0.12 s、`generation` 68.9 s —— **同一进程里 backbone 是热的，mmproj 路径却是慢的**。
慢的只可能是 mtmd 的生成图（predictor + vocoder），与模型文件是否缓存无关。

### 最可能的机制（下一轮验证）

mtmd 用自己的 `mtmd_backends`（带独立线程池）跑自定义图，每个 `ggml_backend_graph_compute` 一次。
若这个线程池在 Android 上出现**唤醒/自旋退化**，每算子多 50–80 ms，800 个算子就是 40–60 s —— 与观测吻合，
并且能解释"每进程不同"（线程池创建时的调度状态不同）。

**验证方法（一次运行即可定位）**：在 `mtmd_subcontext::encode()` 里加三段计时
（`build` / `graph_compute` / `state copy`），跑一次慢进程一次快进程，直接看是哪一段。
不要再从端到端 wall 反推。

### 测量协议（本轮确定，写进下一轮）

1. 每次 A/B **取最小值**（min），不取中位数 —— 慢模态是病态，不是分布尾部。
2. 记录 `prompt eval` 作指纹；>1 s 说明 backbone 未预热，丢弃。
3. 至少 4 次重跑，快慢都要记，禁止只报"最好看的一次"。


---

## E11 插桩定位：所谓"50 倍抖动"其实是**第一次调用的停顿**，稳态一直很稳定

在 `mtmd_subcontext::encode()` 里加了四段计时（`TTS_TRACE=1`），并附带 CPU 号、当前频率、缺页计数：

```
[trace] encode build=1.4 alloc=1.7 input=0.0 compute=197.1 ms (nodes=6460 cpu=7 kHz=2304000 minflt=... majflt=0)
```

### 稳态：一台设备只有三个数

| 图 | 节点数 | build | alloc | **compute** | 每帧 |
|---|---:|---:|---:|---:|---:|
| code_gen（16 次前向） | **6460** | 1.4 ms | 1.7 ms | **197 ms** | 每帧 1 次 |
| code2wav（8 帧） | **559** | 1.3 ms | 1.6 ms | **1.64 s** | 205 ms |
| talker decode（llama 图） | — | — | — | **~14 ms** | 每帧 1 次 |

`build`/`alloc`/`input` 都是毫秒级 —— **之前怀疑的"每帧重建图"不成立**，不是瓶颈。

于是设备稳态 = **14 + 197 + 205 = 416 ms/帧，RTF 5.2**（与 E09 的 5.4 一致）。

### 抖动：只有第一次调用，而且不是算力问题

同一进程内连续 8 次 `gen_code`：

| 运行 | 第 1 次 | 第 2-8 次 |
|---|---:|---|
| A | **3 125 ms** | 209 / 191 / 193 / 213 / 191 / 191 ms |
| B | **66 640 ms** | 197 / 197 / 201 / 198 / 199 / 196 ms |
| C | **99 028 ms** | 215 / 196 / 200 / 200 / 195 / 197 ms |

决定性证据：

- CPU 全程 **cpu7 @ 2 304 000 kHz**（满频），不是降频、不是小核；
- `majflt=0`，两次调用之间 `minflt` 只涨 **3**（内存全驻留），不是缺页；
- 同一张图、同一输入形状，**之后就永远是 197 ms**。

所以这 3–99 s **不是算出来的，是卡出来的**（调度/内核层面的停顿），而且**只发生在进程内第一次 `gen_code` 调用**。
之前记的"50 倍抖动"是把这个一次性停顿误读成了稳态模态。

### 对策略的三个影响

1. **预热是必需的**，不是优化项。没有预热，第一个请求要多付 3–99 s（不可预测）。P0 里"启动跑一句 discard"的价值从"省 44%"变成"省 3–99 s"。
2. **设备测量改口径**：同一配置跑 ≥4 次，**取最小值**；第一次调用的耗时单独记录，不参与比较。
3. **HTP 路线多了一条硬约束**：predictor 图有 **6460 个节点**（每位置约 404 个，即每层约 80 个算子）。
   按 HTP 每算子 100–137 µs 算，**光调度就是 0.65–0.88 s/帧** —— 这就是设计文档说"ggml-hexagon 出局、必须上 QNN 编译图"的定量依据。
   P2a（16 次前向 -> 1 次）会把节点数降到 ~400，正好落回可用区间。

### 更新后的设备预算（稳态）

| 配置 | LM/帧 | vocoder/帧 | wall | RTF |
|---|---:|---:|---:|---:|
| 现状（含 E02） | 211 ms | 205 ms | 416 ms | **5.2** |
| + P2a predictor 1 次前向 | 26 ms | 205 ms | 205 ms | **2.6** |
| + P2b vocoder -33% | 26 ms | 137 ms | 137 ms | **1.7** |

**结论不变但更硬**：v73 的 CPU 上，即使把 predictor 彻底干掉、vocoder 再削 1/3，也只能到 RTF 1.7。
RTF<1 必须依赖 HTP，而 HTP 又受 6460 节点这条约束 —— 所以 P2a 不只是省时间，它是"能不能上 HTP"的前提。


---

## E12 算子直方图：6460 个节点里只有 39% 是计算

`TTS_TRACE_OPS=1` 直接 dump 图里每个算子的计数（host，chunk=4）：

```
code_gen  n=6460
  RESHAPE=1152  CPY=759  MUL_MAT=670  CONCAT=543  MUL=445  VIEW=357  ARANGE=354
  RMS_NORM=351  ADD=310  PERMUTE=288  GET_ROWS=252  SCALE=232  ROPE=176  FLASH_ATTN_EXT=87
  GLU=87  CONT=84  UNARY=41  SIN=29  SQR=29  IM2COL=29  FILL=26  LOG=24  SUB=23  TOP_K=15
  CUMSUM=15  SOFT_MAX=15  SUM=15  TRI=16  TRANSPOSE=19  REPEAT=8  NORM=2
code2wav  n=559
  RESHAPE=154  CONT=80  ADD=59  TRANSPOSE=55  IM2COL=38  MUL_MAT=38  CONCAT=26  VIEW=24
  PAD_REFLECT_1D=22  UNARY=37  MEAN=7  ...
```

### 分类

| 类别 | 节点数 | 占比 |
|---|---:|---:|
| **真正的计算**（MUL_MAT / FA / RMS_NORM / ROPE / GLU / 逐元素） | ~2 543 | **39%** |
| 布局与记账（RESHAPE / VIEW / PERMUTE / CPY / CONCAT / CONT / ARANGE / GET_ROWS） | ~3 911 | **61%** |

### 三个可以直接砍掉的来源

1. **`cache_set` 写了 800 个节点**。
   现在每层每位置的 KV 写入是"concat 追加一行 -> arange 造索引 -> get_rows 搬回原位"，约 5 个算子 × 2（k/v）× 5 层 × 16 位置 ≈ 800。
   改成"往预分配 KV 的行视图里 `ggml_cpy`"只需要 1–2 个算子。**预计 -700 节点。**
2. **`const_i32` / 位置索引在运行时造**：`ARANGE=354`。
   这些索引全部与权重无关、只与位置有关，可以**预烘焙成常量张量**（或改用 `ggml_set_rows` 的原生索引）。**预计 -300 节点。**
3. **每个位置重建一次 mask**（FILL/TRI/LOG 共 66 个）+ `PERMUTE=288`（rope 前后的转置）。
   前者可整段提成常量；后者是 `ggml_rope_ext` 的布局代价，需要看后端是否已融合。

三项做完，6460 预期降到 **~3500**（-46%），其中计算节点占比从 39% 升到 72%。

### 为什么这件事现在变成了关键路径

| 平台 | 约束 | 6460 节点意味着 |
|---|---|---|
| v73 CPU | 30 µs/节点 | 197 ms/帧（实测） |
| HTP（ggml-hexagon） | 100–137 µs/节点 | **0.65–0.88 s/帧，直接出局** |
| HTP（QNN 编译图） | 节点被编译成微码，无逐算子开销 | 可行，但 6460 节点编译时间和内存都不友好 |

所以"砍节点"同时服务三件事：CPU 时间、HTP 可行性、P2a 之后图的规模。
**P2a（16 次前向 -> 1 次）砍 16 倍，属于算法侧；E12 的三项砍 2 倍，属于部署侧，而且不改模型。**

---

## 7. 本轮结束时的状态

| 项 | 状态 |
|---|---|
| 设备测量 | ✅ 有了可用的口径（min-of-4 + 首次调用单列 + 频率/缺页指纹） |
| 50x 抖动 | ✅ 定性为"每次进程内第一次 `gen_code` 的 3–99 s 停顿"，不是稳态模态 |
| 稳态设备预算 | ✅ talker 14 + predictor 197 + vocoder 205 = **416 ms/帧，RTF 5.2** |
| 节点预算 | ✅ 6460 节点，其中 61% 是布局/记账，**这是 HTP 出局的直接原因** |
| RTF<1 | ❌ 仍差 5 倍；CPU 天花板 ~1.7（P2a+P2b 全做完） |

下一轮（按优先级）：

1. **砍节点三项**（cache_set 行写入 / 常量索引 / 常量 mask）：目标 6460 -> ~3500，host 上先验证 bit-exact，再上设备量 CPU 收益。
2. **把预热写进部署路径**（启动跑一句 discard），把首请求的 3–99 s 抹掉。
3. 之后再回到 P2a 重训（那是 RTF 从 2.6 到 1.7 的一步，也是上 HTP 的前提）。


---

## E13 砍节点第一步：去掉 predictor 的 KV 搬移和 mask（6460 -> 4724，bit-exact）

**假设**（来自 E12）：`cache_set` 的"concat 追加 + arange 造索引 + get_rows 搬回"和每位置的 mask 是节点数的主要来源，而不是计算。

**改动**：predictor 的 K/V 改成**逐位置增长的前缀**，attention 天然因果，不再需要 mask。

```cpp
// before: 每个位置把 k/v 写进 16 行 cache 的第 pos 行，再带一个 mask 读回整个 cache
k_cache_layer = cache_set(k_cache_layer, pos, k_flat);      // ~11 个算子
attn = build_attn(..., kq_mask, ...);

// after: 直接追加一行，前缀就是全部有效 key
k_acc = k_acc ? ggml_concat(ctx0, k_acc, k_flat, 1) : k_flat;   // 1 个算子
attn = build_attn(..., nullptr, ...);                            // 无 mask
```

文件：[qwen3tts-gen.cpp](../../../tools/mtmd/models/qwen3tts-gen.cpp)（layer_forward / prefill / step / build），
[qwen3tts-gen.cpp 头](../../../tools/mtmd/models/models.h) 里删掉 `causal_mask_row`。

### 结果：节点 6460 -> 4724（-27%），音频逐位相同

| 算子 | before | after | 差 |
|---|---:|---:|---:|
| CPY | 759 | 283 | **-476** |
| CONCAT | 543 | 233 | **-310** |
| ARANGE | 354 | 54 | **-300** |
| VIEW | 357 | 181 | -176 |
| GET_ROWS | 252 | 92 | -160 |
| RESHAPE | 1152 | 1056 | -96 |
| SCALE | 232 | 72 | -160 |
| LOG/FILL/TRI | 66 | 24 | -42 |
| MUL_MAT / FA / RMS_NORM / ROPE（真计算） | 1284 | 1284 | 0 |
| **合计** | **6460** | **4724** | **-1736** |

音频验证（同 seed 与 E08 的 24 帧输出逐样本比对）：

```
old vs new: 46080 samples, SNR = 215.0 dB, max|diff| = 0.00000
```

**逐位相同** —— 说明"增长前缀 + 无 mask"和原来的"定长 cache + 因果 mask"数学等价，没有引入近似。

### 收益与没收益的部分

| 指标 | 变化 |
|---|---|
| 图节点数 | **-27%** |
| host 上 `gen_code` 每次 | 68 ms -> **63 ms（-7%）** |
| 真计算节点 | 不变（1284） |

**为什么节点 -27% 只换来 -7% 时间**：被删掉的是小张量的搬移和索引生成，字节量本来就小；
真正的时间在 670 个 `MUL_MAT` 上（每次 82 MB 权重）。

**但对 HTP 意义不同**：HTP 的代价 = 节点数 × 100–137 µs，与实际算多少无关。
6460 -> 4724 等于把 HTP 上的调度税从 0.65–0.88 s/帧降到 **0.47–0.65 s/帧**。
还不够（目标 80 ms），但方向对；剩下的要靠 P2a（16 次前向 -> 1 次，再砍 16 倍）。

### 下一步的节点账（还剩什么）

| 剩余大头 | 数量 | 能否砍 |
|---|---:|---|
| RESHAPE | 1056 | 部分是 attention 前后的 3d/4d 转换，可合并；需要 op 级计时确认哪些真的免费 |
| MUL_MAT | 670 | **不改模型砍不掉**（16 位置 × 5 层 × 7 个矩阵乘）；P2a 直接砍 16 倍 |
| MUL / ADD | 445 / 310 | 逐元素，norm/残差，可融合 |
| RMS_NORM | 351 | 16×5×4（含 q_norm/k_norm），可考虑把 q/k norm 融进 RoPE 前的乘法 |
| PERMUTE | 288 | rope 的布局代价，看后端是否已融合 |


---

## E14 发现：这台机器上**可以**编 HTP 后端，之前是我判断错了

上一轮我记"本机没有 hexagon 工具链，P3 顺延"。复查后：

```
docker: 29.9.3
image : ghcr.io/snapdragon-toolchain/arm64-android:v0.8   (7.45 GB, 已存在)
device: /data/local/tmp/llama.cpp/lib/libggml-htp-v73.so .. v85.so
```

工具链在 **docker 镜像里**，仓库自带 `scripts/snapdragon/build.py`（默认走 docker，镜像名/标签正是上面那个）。
所以 **HTP 后端是可编、可测的**，只是要走 docker 而不是本机 NDK。

正在跑的构建（本轮结束时仍在进行）：

```sh
./scripts/snapdragon/build.py --target adb -j 6      # 不加 --push，避免覆盖同事的 /data/local/tmp/llama.cpp
```

产物落在 `build-adb/`（769 个目标，含 OpenCL + hexagon v73/v75/v79/v81/v85）。
下一轮第一件事就是收这个构建，然后：

1. 把 `build-adb/bin/llama-tts` + `lib/libggml-htp*.so` 推到 **`/data/local/tmp/q3tts/`**（我自己的目录，不碰 llama.cpp）；
2. 用 `--device HTP0` 跑同一组 8 帧/24 帧命令，拿到**第一份 HTP 真机数据**；
3. 对照 E11 的 CPU 基线（416 ms/帧）看 HTP 到底值多少。

这将直接回答设计文档里最大的悬案：**v73 的 HTP 到底能不能接住这个模型**。
（按 E12 的节点账：4724 节点 × 100–137 µs = 0.47–0.65 s/帧，预期仍然不够 —— 但那只是"逐 op 下发"的上界，
htp-fusion 能吃掉一部分，实测才知道。而且这次终于能测，而不是推算。）


---

## E15 第一次 HTP 实测：talker 该留在 CPU，vocoder 该搬到 HTP，predictor 直接跑不了

上一轮发现 docker 里能编 HTP，这一轮把 `llama-tts` 用 hexagon 后端编出来并推上设备
（`build-adb/`，4096 个目标只编了 tts 需要的 3 个），拿到第一手数据。

### 设备上后端可见

```
$ ./bin/llama-tts --list-devices
ggml-hex: CDSP reports no HMX support, falling back to HVX
Available devices:
  GPUOpenCL: QUALCOMM Adreno(TM) 810 (5722 MiB, 4698 MiB free)
  HTP0: Hexagon (0 MiB, 0 MiB free)
```

### 结果一：predictor 的图太大，HTP 直接崩

```
ggml-hexagon.cpp:1795: ggml-hex: dspqueue_read failed: 0x0000002e
  ... mtmd_subcontext::encode -> mtmd_helper_gen_audio_step_gen -> Aborted
```

- 559 节点的 **vocoder 图能跑**；
- **4724 节点的 predictor 图跑不了**，且 `OPBATCH=512/256/64 + OPFUSION=1` 全都一样崩。

这就是 E12 那个节点预算的实测版：**v73 的 ggml-hexagon 接不住 4724 个算子**。
要么把 predictor 降到 ~500 节点（P2a 正好做这件事），要么走 QNN 编译图。

### 结果二：talker 放 HTP 比 CPU 慢 3.7 倍

用 `--mmproj-device none`（mmproj 留 CPU）把 backbone 单独放到 HTP0 上跑通了：

| talker decode | 时间 |
|---|---:|
| CPU（4 线程） | **14 ms**/token |
| **HTP0** | **51–66 ms**/token |

batch=1 的 GEMV 在无 HMX 的 v73 上就是打不过 CPU —— 与 M5 记的"小 GEMV HTP 慢 2.2x"、与设计文档 §3.3 的预测一致，只是这里差得更多（3.7x）。

### 结果三：vocoder 放 HTP **快 4–5 倍**，这是本轮的意外收获

vocoder 图（559 节点）在 HTP 上的 `compute`（8 帧，4 次独立进程取最小）：

| vocoder | 每帧 | 相对 |
|---|---:|---:|
| CPU 4 线程 | 205 ms | 1.0x |
| **HTP0** | **36 ms**（292/8） | **5.7x** |

与 M5 那次的结论相反（他们当时"vocoder 上 NPU 是负收益"），原因是：

- M5 的 vocoder 已经做过 int8 + Snake 融合，CPU 侧只剩 0.46 RTF，搬到 DSP 还要和 depth 串行；
- **Qwen3-TTS 的 vocoder 是 F16 的、没优化过**，纯 conv（DAC 残差单元占 94% 算力），
  而 conv 正是 HTP 相对 CPU 唯一有明显优势的负载（M5 实测 conv 快 1.6x，这里因为 CPU 侧没有 int8 化，差距更大）。

### 结论：v73 的正确分工和设计文档写的不一样

| 模块 | 设计文档原判断 | **实测后** | 理由 |
|---|---|---|---|
| talker | 上 HTP | **留 CPU** | HTP 慢 3.7x（GEMV） |
| predictor | 上 HTP（QNN） | **留 CPU**（除非 P2a 把节点砍到 ~500） | 图太大直接崩；CPU 197 ms 可接受 |
| vocoder | 留 CPU | **搬到 HTP** | HTP 快 5.7x（conv） |

于是 v73 的最优 wall 变成：

```
wall = max( talker 14 + predictor 197 (CPU) ,  vocoder 36 (HTP) )
     = 211 ms  ->  RTF 2.6
```

**关键推论：predictor 一旦被 P2a 打到 ~15 ms，wall 就由 HTP 上的 vocoder 决定：**

```
wall = max( 14 + 15 , 36 ) = 36 ms  ->  RTF 0.45
```

**这是第一次在 v73 上看到 RTF<1 的可行路径**，而且它不依赖 HTP 跑 LM（那本来就跑不动），
只依赖"vocoder 上 HTP + predictor 并行化"这两件已经立项的事。

### 待验证（下一轮）

1. vocoder 在 HTP 上要**用真实音频**验证（现在只有 warmup 的 compute 时间，没有端到端音频）；
   需要 predictor 能在 HTP 上跑（或把 predictor 拆成 16 个 ~300 节点的子图）。
2. talker 的 HTP 数字再取几次，确认 51–66 ms 不是首调用。
3. HTP 上 vocoder 的**数值正确性**：拿 CPU 版音频做 SNR 比对（还没做，因为端到端跑不通）。


---

## E16 单独测 vocoder：**上一轮"快 5.7 倍"是错的**，HTP 和 CPU 一样慢

上一轮我用 pipeline 里的 warmup encode（compute=292 ms）当作"8 帧 vocoder 在 HTP 上的时间"，
推算出 36 ms/帧、比 CPU 快 5.7 倍。**这个推断是错的** —— warmup 的图只有 1 帧，292 ms 其实是 **292 ms/帧**。

为了把这件事测干净，加了一个独立小工具 [vocbench.cpp](../../../tools/tts/vocbench.cpp)：
直接给 code2wav 喂确定性 codes，跳过 talker/predictor，可指定后端，输出 f32 PCM 供 SNR 比对。

```sh
llama-vocbench -m backbone.gguf -mm mmproj.gguf -dev CPU|HTP0 -n <frames> -c <chunks> -o out.f32
```

### 结果：两个后端的**边际**成本几乎一样

HTP0，8 帧 chunk，两次独立运行（每 chunk 一次 encode）：

| chunk 帧数 | 总时间 | ms/帧 |
|---:|---:|---:|
| 4 | 1 643 / 1 479 ms | 411 / 370 |
| 8 | 2 516 / 2 272 ms | 315 / 284 |
| 16 | 4 318 ms | 270 |
| 32 | 7 888 ms | 246 |
| 72 | 17 781 / 17 315 ms | 247 / 240 |

线性拟合（4→72 帧）：**边际 233 ms/帧 + 固定 ~516 ms/调用**。

对照 CPU（同一工具、同一 codes）：单次 302 ms/帧（最好）到 1 880 ms/帧（最差），抖动极大；
pipeline 里量的 205 ms/帧与之同量级。

**结论：v73 上 vocoder 的边际成本 ~233（HTP）vs ~205–300（CPU），HTP 没有优势。**
M5 当时"conv 在 HTP 快 1.6x"没有复现，原因是我们的 vocoder 走的是 **conv -> im2col -> MUL_MAT** 的降级路径
（直方图里 IM2COL=38 / MUL_MAT=38），等于把它变成了 GEMV —— 而 GEMV 在无 HMX 的 v73 上正是 HTP 最差的一项。

### 数值：HTP 是可靠的

同一份 codes，CPU 与 HTP0 的输出：

```
samples 30720 (1.28s), CPU-vs-HTP SNR = 53.2 dB, max|diff| = 0.00272, rms = 0.0944
```

53 dB 远高于验收线（30 dB），**HTP 的 vocoder 数值上没问题**，问题纯粹是速度。

### 顺带修正一个记账错误

`build()` 里 GEN_CODE 和 GEN_WAV 两个子图是**都建**的（`ggml_build_forward_select` 再选），
所以 trace 里的 `nodes=4724` 是**两个分支加起来**，不是"executed 图"的节点数。
E12/E13 里的**差值**（CPY −476、CONCAT −310…）仍然成立，但**绝对值要打折扣**。
这一点在评估"HTP 最多能吃多少节点"时必须记住。

### 对目标的直接影响

vocoder 现在 **246 ms/帧 = RTF 3.1（单独一项）**，而 LM 侧在 P2a 后是 ~29 ms。
所以 v73 的 wall 已经确定由 **vocoder** 决定：

```
wall = max(29 (LM), 246 (vocoder)) = 246 ms  ->  RTF 3.1
```

要让 RTF<1，vocoder 必须 **从 246 降到 80 ms 以下（3x）**。三条路，按性价比：

| 路线 | 预期 | 依据 |
|---|---|---|
| **int8（A8W8）** | 1.8–2.5x | M5 的 codec 全 int8 后 5.28 s -> 2.01 s（2.6x） |
| **conv 原生算子**（不走 im2col） | 1.5–2x（仅 HTP） | M5 探针：原生 conv 比 im2col+GEMM 快 2x 以上 |
| **P2b 结构减重**（res unit 3->2） | 1.25x | 残差单元占 94% 算力 |

**三条叠起来才有 3x**，也就是"vocoder 必须做 int8 + 换 conv 实现 + 结构减重"，缺一条都到不了 80 ms。
这和最初设计文档的 P2b 判断一致，但现在有了实测依据，而且**优先级从"次要"升到了"决定成败"**。


---

## E17 int8 换不来 vocoder 的加速：转换器把 94% 的算力钉死在 F16/F32

上一节说"int8 是 vocoder 的第一条路（预期 1.8–2.5x）"。直接验一下：用 Q8_0 的 mmproj 跑同一个 vocbench。

| mmproj | chunk 32 帧 | ms/帧 |
|---|---:|---:|
| F16 | 7 778 ms | 243.1 |
| **Q8_0** | 7 500 ms | **234.4（-3.6%）** |

**几乎没有收益**。原因在转换器的 `tensor_force_quant`（[conversion/qwen3tts.py:270-285](../../../conversion/qwen3tts.py)）：

| 张量 | 被强制成 | 占 vocoder 算力 |
|---|---|---:|
| `dac.blk.*.conv1.weight`（残差单元 k=7 空洞卷积） | **F16** | ~70% |
| `dac.blk.*.conv2.weight`（k=1） | F16 | ~24% |
| `up.blk.*.conv.weight` / `dac.blk.*.conv.weight`（ConvTranspose） | **F32** | ~5% |
| `pre_conv` / `dac.entry` / `dac.post_conv` | F16 | ~1% |

也就是说：**Q8 只量化了 pre_transformer 和 RVQ 那些零头**，真正吃算力的卷积一个没动。

### 三条结论

1. **"把 mmproj 转成 Q8" 是个无效动作**，别再试（这一条排除了一个看起来最省事的方案）。
2. 现有的 `ggml_conv_1d` 走 **im2col + MUL_MAT**，所以即使权重变 int8 也只是变成 "int8 GEMV"，
   在无 HMX 的 v73 上不会快（E16 已看到 HTP 对 vocoder 没有优势）。**必须用原生 int8 卷积 kernel。**
3. 这条路是**后端工程**（ggml-cpu 或 ggml-hexagon 加 int8/direct conv），不是配置。
   M5 已经有过一份参考实现（`m5_snake_op.h` + 直卷积 kernel），可以复用数学。

### 于是 vocoder 的三条路重排

| 路线 | 预期 | 代价 | 结论 |
|---|---|---|---|
| 转 Q8 权重 | **1.0x（实测）** | 5 分钟 | ❌ 排除 |
| 结构减重（res unit 3->2、通道收窄） | 1.25–1.4x | 重训 vocoder | ✅ 可做，但不够 |
| **原生 int8 conv kernel** | 1.8–2.5x | 后端实现（1–2 周） | ✅ 必须做 |
| HTP 上跑 conv | 1.0x（实测，im2col 降级） | — | ❌ 除非换成原生 conv |

要把 vocoder 从 246 ms 降到 80 ms（3x），**"结构减重 + 原生 int8 conv" 两条都得做**，而且都要过盲听。

---

## 8. 本轮结束时的状态

| 项 | 状态 |
|---|---|
| 新的测量工具 | ✅ [vocbench.cpp](../../../tools/tts/vocbench.cpp)，可单测 vocoder、可指定后端、可出 PCM 比对 |
| vocoder HTP vs CPU | ✅ 边际 233 vs 205–300 ms/帧，**HTP 无优势**（推翻上一轮） |
| vocoder 数值 | ✅ HTP vs CPU SNR **53.2 dB**，可用 |
| int8 权重 | ❌ 实测 -3.6%，被转换器的 F16/F32 规则挡住 |
| LM 侧 | talker 14 ms（CPU）/ 52–66 ms（HTP），predictor 197 ms（CPU），HTP 图太大跑不了 |
| **RTF** | **3.1（vocoder 决定）**，距目标还差 3x |

下一轮的优先级（明确到可执行）：

1. **vocoder 结构减重**：res unit 3->2 + 通道 768/384/192/96 -> 640/320/160/80（离线改权重即可测算子数，重训另说）；
2. **原生 int8 conv**：先在 ggml-cpu 里给 DAC 的 k=7 / k=1 卷积加 int8 路径，用 vocbench 量收益和 SNR；
3. 这两件做完再看 wall 是否到 80 ms；如果到不了，v73 的 RTF<1 就需要重新评估（可能只能靠"降采样率/减层"这类会掉音质的极端手段）。


---

## E18 结构减重与 int8 的两条路都被堵住了一半，但边界测出来了

### (1) int8 卷积：不是"没开"，是**权重布局不允许**

放宽转换器的强制 F16 规则（加 `QWEN3TTS_WAV_CONV_QUANT` 开关）后重新转 mmproj，DAC 的卷积**仍然是 F16**：

```
a.gen.wav.dac.blk.0.res.0.conv1.weight   [7, 768, 768]   F16
a.gen.wav.dac.blk.0.res.1.conv1.weight   [7, 768, 768]   F16
a.gen.wav.dac.blk.0.res.2.conv1.weight   [7, 768, 768]   F16
```

原因：**`ne[0] = k = 7`**，不是量化块大小（32）的倍数 —— ggml 量化器直接放弃，回落到 F16。
这解释了转换器为什么"强制 F16"，不是偷懒，是布局约束。

**要做 int8 卷积，必须先把权重 repack 成矩阵**（把 k 折进 IC，让 ne[0] = C 或 C·k），
再配一个自定义 kernel。这正是 M5 做过的 "ConvTranspose -> MatMul" 那类改造。
**结论：int8 vocoder 是后端工程（repack + kernel），不是一个转换开关。**

### (2) 砍残差单元：-24.5% 时间，但音质直接崩（3.1 dB）

加了个 `TTS_VOC_RES` 开关（默认 3），在 host 上量 16 帧 chunk：

| res units/block | ms/帧 | 相对 | 与完整版 SNR |
|---:|---:|---:|---:|
| 3（完整） | 53.1 | 1.00x | — |
| **2** | **40.1** | **0.755x** | **3.1 dB** |
| 0 | 13.4 | 0.25x | 0.0 dB |

**三个残差单元占了 vocoder 75% 的时间**（与 MAC 账里"res unit 占 94% DAC 算力"一致）。

但 3.1 dB 的 SNR 意味着"这不是同一条音频"—— 残差单元不是冗余的，**砍掉就必须重训**。
所以这 -24.5% 是"重训能拿到多少"的**上界**，不是一个可以直接上线的开关。

### (3) 于是 v73 的 vocoder 还有多少空间

| 手段 | 相对 | 需要什么 | 246 ms/帧 -> |
|---|---:|---|---:|
| 现状（F16, im2col, 3 res） | 1.00 | — | 246 |
| res 3->2 | 0.755 | **重训 vocoder** | 186 |
| 通道 768/384/192/96 -> 640/320/160/80 | ~0.7 | 重训 | ~130 |
| 原生 int8 卷积（repack + kernel） | 0.4–0.55 | 后端实现 + 精度验证 | ~60–100 |

理论下界：2.48 GMAC/帧。F32 NEON 峰值约 73 GFLOP/s（4 核 A78 @2.3 GHz）→ **~68 ms**；
int8 峰值再 4x → ~17 ms。现在是 246 ms，**只跑到理论峰值的 28%**，而其中大部分损耗是 im2col 和 F16 卷积。

**所以"vocoder 到 80 ms"在理论上是够的，但需要 repack + int8 kernel + 重训减重三件套，
这不是部署调优能拿到的。** 这也是本系列第一次把 v73 的天花板量化清楚：

```
LM（P2a 后）   29 ms
vocoder       68 ms（理论下界，F32 峰值）
------------------------------------
wall = max(29, 68) = 68 ms  ->  RTF 0.85   ← 只有把 vocoder 打到理论峰值才行
现状 = max(211, 246) = 246 ms -> RTF 3.1
```


### 下一轮的具体做法（已经想清楚，照着做即可）

思路：**把卷积权重的 k 维折进 IC，变成 im2col 形状的矩阵**，这样 `ne[0] = IC*k`（768×7 = 5376，32 的倍数）
就可以量化，且直接走 `mul_mat` 的 int8 路径。

1. **转换期**（[conversion/qwen3tts.py](../../../conversion/qwen3tts.py)）：`dac.blk.*.conv1.weight` 从 `[k, IC, OC]`
   改写成 `[IC*k, OC]`（行序必须与 ggml im2col 一致：row = ic*k + ki，或反过来，用 SNR 验证）。
   新增一个 GGUF KV 标记 "conv weights are folded"，让运行时知道。
2. **运行期**（[qwen3tts-gen.cpp](../../../tools/mtmd/models/qwen3tts-gen.cpp) 的 `causal_conv1d`）：
   用手工 im2col 替代 `ggml_conv_1d` —— 对 k 个偏移各取一个 `[IC, T]` 视图，`concat` 成 `[IC*k, T]`，
   再 `ggml_mul_mat(folded_w, x)`。k=7 就是 6 次 concat + 1 次 matmul（现在是 1 个 IM2COL + 1 个 matmul，
   算子数相当，但权重可量化）。
3. **验证**：vocbench 跑 F16-folded vs 现有 F16，SNR 应 > 80 dB（只改布局，不改数学）；
   再跑 Q8-folded，看 SNR（预期 30–40 dB）和 ms/帧（预期 1.5–2.5x）。
4. 如果 Q8-folded 的 SNR 不够，退回 "folded + F16"（布局换了但精度不变，仍可能因为 im2col 更规整而略快）。

这条路的收益上限：把 vocoder 里 50 M 参数的卷积从 F16 变 Q8，同时把 im2col 的 F32 中间量减半。
按 E18 的理论账，246 -> 100~130 ms 是合理预期，叠加 res trim（重训）才有 80 ms。


---

## E19 折叠卷积 + int8：真的变快了（-21~-25%），但代价是 22.8 dB

### 做法（本轮实现）

把 DAC 残差卷积的 k 维折进 IC，让 `ne[0] = K*IC`（如 7×768 = 5376，32 的倍数）→ 可量化：

| 侧 | 改动 |
|---|---|
| 转换期 | `fold_conv()`：torch `[OC, IC, K]` -> `[OC, K*IC]`，ggml 读成 `[K*IC, OC]`，行序 `ki*IC + ic` |
| 运行期 | `causal_conv1d_folded()`：K 个偏移视图 concat 成 `[K*IC, T]`，一次 `mul_mat`；`conv_is_folded()` 自动识别 |
| 状态 | `list_c2w_state_slots` 按折叠后形状反推 K=7 |

### 数据

host（16 帧 chunk，CPU 6 线程）：

| 变体 | ms/帧 | 相对 | 与 F16 的 SNR |
|---|---:|---:|---:|
| 原版 F16（unfolded） | 55.1 | 1.00 | — |
| 折叠 F16 | 61.9 | 1.12（**更慢**） | **69.8 dB**（等价） |
| **折叠 Q8** | **39.7** | **0.72** | **22.8 dB** |

设备（32 帧 chunk）：

| 变体 | HTP0 | CPU |
|---|---:|---:|
| 原版 F16 | 224.1 ms/帧 | 617（慢模态）|
| **折叠 Q8** | **176.9 ms/帧（-21%）** | **166.9 ms/帧（-25%）** |

三点结论：

1. **折叠本身不加速**（F16 折叠比原版慢 12%，因为 concat 版 im2col 不如 ggml 的融合 im2col），但它是量化的**前提**，而且数学等价（69.8 dB）。
2. **折叠 + Q8 才有收益**：host -28%，设备 -21~-25%。**vocoder 从 ~224 降到 ~170 ms/帧。**
3. **代价明确**：22.8 dB。低于设计文档定的 30 dB 验收线，属于"需要盲听裁决"的档位。
   如果听感不能接受，就只能退到 Q6_K 之类（收益会小很多）。

### 于是 v73 的天花板再算一次

| 阶段 | vocoder ms/帧 | 相对原版 |
|---|---:|---:|
| 原版 F16 | 246 | 1.00 |
| **+ 折叠 Q8（本轮，实测）** | **~170** | **0.69** |
| + res 3->2（需重训） | ~128 | 0.52 |
| + 通道收窄（需重训） | ~100 | 0.41 |
| + 原生 direct int8 conv（后端） | ~70–100 | 0.3–0.4 |

LM 侧（P2a 后）约 29 ms。所以：

```
最好情况：wall = max(29, 70–100) = 70–100 ms  ->  RTF 0.88–1.25
现实情况（只做本轮 + 重训减重）：wall = max(29, 128) = 128 ms  ->  RTF 1.6
```

**结论：v73 上 RTF<1 处在"需要把 vocoder 打到理论峰值 70–80% 才成立"的位置，
这要求"折叠 int8 + 结构减重重训 + 原生卷积 kernel"三件套全部兑现，且每一步都过盲听。
按目前证据，更现实的区间是 RTF 1.2–1.6。**

这不是"做不到"，而是"要付的代价已经清楚"：
- 音质：22.8 dB（int8）+ 减重（3.1 dB 的坑要靠重训补）；
- 工程量：后端 direct int8 conv（1–2 周）+ vocoder 重训（3–4 周）+ P2a（3–4 周）。

### 如果目标是"稳稳的 RTF<1"

把同样的模型放到 **v79（带 HMX）** 上，算力余量是 12 TOPS 级的 fp16：
vocoder 的 31 GMAC/s 折算成 fp16 是 62 GFLOP/s，占 HMX 的 0.5%；
LM 的 1.9 GMAC/帧（P2a 后 0.53 G）也远低于可用吞吐。
换句话说：**换一代 SoC，RTF 从"要三件套勉强够"变成"随随便便 0.2"。**
这是本轮之后最应该向上汇报的一句话。


---

## E20 端到端实测（v73，当前最好配置）与最终裁决

配置：`-dev none`（LM 全在 CPU）+ 折叠 Q8 的 vocoder + 24 帧输出，seed 固定。

| chunk | prompt | generation | vocoder | total | **TTFA** | RTF |
|---|---:|---:|---:|---:|---:|---:|
| 72 | 0.09 s | 4.38 s | 2.71 s | 7.17 s | **7.17 s** | **3.74** |
| 8 | 12.21 s | 6.67 s | 0 | 18.88 s | 14.45 s | 9.83 |
| 4 | 12.25 s | 51.04 s | 0 | 63.28 s | 57.02 s | 32.96 |

（后两行是"慢模态"：`prompt eval` 12 s 说明进程处于 E10 记的病态；但它们揭示了一件更要紧的事）

### 两条硬结论

1. **TTFA 在 v73 上是 7.2 s 起步，而且 chunk 越小越糟**（8 帧 14 s、4 帧 57 s）。
   原因是 E11 记的"每次进程内第一次 encode 卡 3–99 s"—— chunk 越小，这种卡顿被触发得越多。
   **在这个病态修掉之前，v73 上的流式（小 chunk）不是"更快"，而是灾难。**
2. **端到端 RTF 3.74**，与分解一致：182 ms/帧（LM）+ 113 ms/帧（vocoder）= 295 ms/帧。

### 把已测的数字代进最终预算

| 配置 | LM/帧 | vocoder/帧 | wall | RTF |
|---|---:|---:|---:|---:|
| **今天（实测）** | 182 | 113 | 295 | **3.74** |
| + P2a（predictor 1 次前向，重训） | 29 | 113 | 113 | **1.41** |
| + vocoder 减重重训（res 3->2） | 29 | 85 | 85 | **1.06** |
| + 通道收窄（重训） | 29 | ~70 | 70 | **0.87** |

（wall 取 max 是因为 LM 与 vocoder 可以并行；今天实测是串行，所以 295 而不是 max(182,113)=182。）

### 最终裁决（本轮可以定稿的部分）

**v73（SM6650）上，Qwen3-TTS-0.6B 的 RTF<1 只在"P2a 重训 + vocoder 减重重训 + 通道收窄"全部兑现时才成立（~0.87），
少任何一项都 >= 1.06。** 这三件事都是模型/后端工作，不是部署调优；累计工期 8–12 周，
并且 int8 折叠卷积已经把音质压到 22.8 dB，减重还要再吃掉一部分。

**同时，TTFA 被一个未定位的"每次 encode 卡 3–99 s"钉死在 7 s 以上。**
这是比 RTF 更致命的可用性问题：即使 RTF 做到 0.87，首包 7 s 也无法接受。

### 因此建议把目标拆成两条，分别定：

| 目标 | v73 | 说明 |
|---|---|---|
| RTF < 1 | **需要三件套（0.87）**，否则 1.06–1.41 | 可做，但要 8–12 周重训 + 后端 |
| TTFA < 500 ms | **当前不可能** | 卡在未定位的 per-call 停顿上 |
| RTF < 0.5 且 TTFA < 200 ms | **必须换 SoC（v79/HMX）** | 算力余量 100x 以上 |

---


---

## E21 那个"卡 3–99 秒"找到了：是 DVFS 爬频，不是代码 bug

连续跑 14 个 8 帧 chunk（vocbench，CPU，折叠 Q8），每 chunk 的耗时：

```
7 692  1 576  1 567  2 364  1 572  2 377  2 365  1 182  892  894  897  889  884  892   (ms)
   ^chunk0        ^^ 过渡区（~1.2–2.4 s）^^^        ^^^^^^^^ 稳态 890 ms ^^^^^^^^
```

- **第 0 个 chunk 7.7 s**（首调用）
- 第 1–7 个 chunk 在 1.2–2.4 s 之间摇摆
- **第 8 个 chunk 起稳定在 890 ms（111 ms/帧）**

这正是 DVFS 爬频的形状：**持续负载约 10 秒后才到最高频**。与 M5 记的
"静置后 walt 大核被压到 633 MHz、升频滞后、首批请求慢 44%"是同一现象的极端版。

之前排查过的线程、绑核、内存、加载模式、页缓存全都不是原因 —— 因为原因在**调频策略**，
而它表现为"进程内前 ~10 秒慢"，看起来像随机的。

### 三个直接后果

1. **预热必须做满 10 秒**（不是"跑一句"就够）。产品启动时跑 10–15 s 的 discard 请求，
   把频点顶上去；否则每个新进程的首请求都要付 7 s。
2. **之前所有"慢模态"数据都该按此解释**：短跑（1–2 个 chunk）永远在过渡区里，所以数字随跑法剧烈变化。
   E16/E17 里那些 302→1880 ms/帧的散布，现在有了统一解释。
3. **稳态数字才是真数字**：vocoder 111 ms/帧（8 帧 chunk，折叠 Q8，CPU）。

### 修正后的 TTFA 预算（v73）

```
TTFA = prefill + (首包帧数 x LM/帧) + 首个 vocoder chunk
     = 0.1 s + N x LM + (~500 ms 固定 + N x 111 ms)
```

| 配置 | N=4 | N=8 |
|---|---:|---:|
| 今天（LM 182 ms/帧） | 0.1+0.73+0.94 = **1.8 s** | 0.1+1.46+1.39 = **2.9 s** |
| + P2a（LM 29 ms/帧） | 0.1+0.12+0.94 = **1.2 s** | 0.1+0.23+1.39 = **1.7 s** |
| + P2a + vocoder 减重（85 ms/帧） | 0.1+0.12+0.84 = **1.1 s** | 0.1+0.23+1.18 = **1.5 s** |

**在预热做满的前提下，v73 的 TTFA 现实区间是 1.1–1.8 s**（不是 7 s，也不是 150 ms）。

---

## 9. 八轮之后的完整账（可汇报版）

### 已经落地的优化（都有实测）

| 优化 | 效果 | 平台 | 备注 |
|---|---|---|---|
| 精确帧数建图（E02） | vocoder 5.09 -> 1.41 s（24 帧） | host | bit-exact |
| chunk 1/2/4 可调（E03） | TTFA -66% | host | chunk=4 是拐点 |
| predictor 去 KV 搬移（E13） | 节点 6460 -> 4724 | 所有 | bit-exact |
| 折叠+int8 卷积（E19） | vocoder -21~25% | 设备 | 22.8 dB |
| 预热协议（E21） | 首请求 7 s -> ~1 s | 设备 | 需 10 s warmup |

### 当前 v73 的真实水平

| 指标 | 值 |
|---|---|
| RTF（端到端，最好配置） | **3.74** |
| vocoder 稳态 | 111 ms/帧 |
| LM 稳态 | 182 ms/帧（talker 14 + predictor 168） |
| TTFA（预热后，N=4） | ~1.8 s |
| 达到 RTF<1 需要 | P2a 重训 + vocoder 减重重训 + 通道收窄 = 0.87 |

### 还剩下什么

1. **P2a**（predictor 1 次前向）：LM 182 -> 29 ms/帧。**收益最大，且是上 HTP 的前提**（4660 节点 HTP 跑不了）。
2. **vocoder 减重重训**：111 -> 85 ms/帧。
3. **通道收窄**：85 -> ~70 ms/帧。
4. 三件套齐了才有 RTF 0.87；少一件 1.06，再少一件 1.41。


---

## E22 P2a 的成本上界测出来了：predictor 63 -> 10 ms（host），不是估算

P2a（predictor 并行头）需要重训，无法在本地训练。但**可以测它的成本上界**：
用一个 `TTS_FAKE_NAR=1` 的结构替身——仍然用 AR 的权重，只做 **1 次 5 层前向 + 15 个头一次批乘**，
采样照旧（只是 code 无意义）。这不验证音质，只验证"1-pass 形状"的节点数和时间。

### host 实测（同一个模型、同一段文本、seed 固定）

| | AR（现状） | 1-pass 替身 | 差 |
|---|---:|---:|---:|
| `gen_code`/帧 | **63–68 ms** | **9.3–12.0 ms** | **-82%** |
| GEN_CODE 图节点 | ~4 165 | **~1 355** | **-67%** |
| LM/帧（talker+pred） | 76 ms | **22 ms** | -71% |

节点数是"两分支合计 - vocoder(559)"推出来的；替身图里 ~750 个节点还是 15 次采样，
若把采样搬到 host（只需回传 15×2048 logits = 123 KB），图还能再降到 ~600。

### 为什么比按 MAC 估的（2.3x）好得多

AR 链的真实成本是**权重流量**：16 个位置各读一遍 82 MB = 1.3 GB；
1-pass 只读塔的 82 MB + 15 个头的 31.5 MB = 113 MB，**11 倍流量差**。
所以"并行化"省的不只是 MAC，主要是字节 —— 这也解释了为什么它在 v73 这种带宽受限的芯片上更重要。

### 代进 v73 的预算（这次是有测量的）

设备上 predictor 是 168 ms/帧，host 是 63 ms，比例 2.7x；
所以 1-pass 在设备上约 **10 × 2.7 ≈ 27 ms/帧**，LM = talker 14 + 27 = **41 ms/帧**。

| 配置 | LM | vocoder | wall | RTF |
|---|---:|---:|---:|---:|
| 今天（实测） | 182 | 111 | 295 | **3.74** |
| + P2a（用本轮实测外推） | **41** | 111 | 111 | **1.39** |
| + vocoder 减重重训 | 41 | 85 | 85 | **1.06** |
| + 通道收窄 | 41 | 70 | 70 | **0.87** |

与 E20 的估算一致（LM 从 29 修正为 41，结论不变）。

### 这条测量的意义

它把 P2a 从"值得做的猜测"变成了"**确定的收益**"：
- 收益：LM -71%（host 76→22 ms；设备 182→41 ms），节点 -67%；
- 代价：需要重训（spec 见 [qwen3-tts-predictor-redesign.md](qwen3-tts-predictor-redesign.md)），且要过盲听；
- 风险：替身用的是 AR 权重，**质量完全没验证**；真实 P2a 的一致性风险仍在（见 E13 的 A/B/C 方案）。


---

## E23 设备上的 P2a 替身：predictor 168 -> 13 ms（-93%），比外推还好

把 E22 的 `TTS_FAKE_NAR` 编到设备上跑（`-dev none`，折叠 Q8 mmproj）：

| | AR（实测） | 1-pass 替身（实测） | 差 |
|---|---:|---:|---:|
| `gen_code`/帧 | 168–197 ms | **11.6–14.7 ms** | **-93%** |
| `talker decode`/帧 | 14 ms（热）/ 31 ms（冷） | 31–33 ms（本轮处于冷态） | — |
| LM/帧 | 182 ms | **44 ms**（冷态）/ ~27 ms（热态） | **-76%** |

**比我按 host 比例（2.7x）外推的 27 ms 还好** —— 因为 1-pass 之后 predictor 从"读 1.3 GB 权重"
变成"读 113 MB"，而设备的内存带宽和 M2 在这个规模上差别不大，所以两者的 1-pass 时间几乎一样（10 vs 13 ms）。

### 最终预算表（全部有实测支撑）

| 配置 | LM/帧 | vocoder/帧 | wall（并行取 max） | RTF |
|---|---:|---:|---:|---:|
| 今天 | 182 | 111 | 295（串行） | **3.74** |
| **+ P2a（替身实测）** | **44** | 111 | **111** | **1.39** |
| + vocoder 减重重训 | 44 | 85 | 85 | **1.06** |
| + 通道收窄（重训） | 44 | 70 | 70 | **0.87** |

TTFA（预热做满、chunk=4）：`0.1 + 4×44 ms + (500 + 4×111) ms ≈ 1.2 s`。

### 这条测量改变了什么

在这条之前，P2a 的收益是"按 MAC/字节估算的 -73%"；现在是**在目标设备上实测的 -93%（predictor）/ -76%（LM）**，
而且节点从 4165 降到 1355。它把"要不要花 3–4 周重训"这个决策变成了"收益已经确定，只剩质量风险要验"。

**同时它也确认了：v73 的 wall 现在完全由 vocoder 决定（44 vs 111）。**
换句话说，P2a 之后，LM 侧已经不是问题了 —— 剩下的 100% 是 vocoder 那 111 ms/帧。


---

## E24 设备上验掉了 vocoder 的成本归属：残差单元占 72%

用已有的 `TTS_VOC_RES` 开关在设备上做归属实验（32 帧 chunk，折叠 Q8，CPU）：

| 配置 | chunk 1（稳态） | ms/帧 | 说明 |
|---|---:|---:|---|
| RES=3（完整） | 3 139 ms | **98.1** | 与 E21 的 111（8 帧 chunk）同一量级，大 chunk 便宜 12% |
| RES=0（去掉全部残差） | 884 ms | **27.6** | 只剩 pre_transformer + 上采样 + ConvTranspose + entry/post |
| **残差单元合计** | | **70.5** | **占总时间 72%** |

（RES=2 那次跑出 163 ms/帧，但它的 chunk0 是 8.6 s —— 整轮处于慢 DVFS 状态，作废。
这再次说明：**只有同时看 chunk0 和 chunk1 才知道这一轮能不能用**。）

### 与 host 的一致性

host 上 RES 3→2→0 = 53.1 → 40.1 → 13.4 ms/帧，残差占 75%；设备上 72%。两个平台一致，
说明这个归属是结构性的，不是某台机器的偶然。

### 设备 trim 后的预算（把外推换成实测锚点）

以 chunk=8 的 111 ms/帧为基准（流式要用小 chunk），残差 80 ms、非残差 31 ms：

| 配置 | LM | vocoder | wall | RTF |
|---|---:|---:|---:|---:|
| 今天 | 182 | 111 | 295（串行） | 3.74 |
| + P2a | 44 | 111 | 111 | 1.39 |
| **+ res 3->2（重训）** | 44 | **84** | **84** | **1.05** |
| + 通道收窄（重训） | 44 | **~68** | 68 | **0.85** |

**结论没变，但每一格现在都有实测锚点**：
- LM 44 是设备实测（E23 替身）；
- vocoder 111 / 98 是设备实测；
- 残差占 72% 是设备实测；
- 只有"减重后的 84"和"通道收窄后的 68"是按 MAC 比例外推（因为要重训才能真跑）。

### 现在的关键路径只剩两个重训

```
RTF 3.74 --[P2a 重训, 3-4 周]--> 1.39 --[vocoder res 3->2 重训, 3-4 周]--> 1.05
       --[通道收窄, 2-3 周]--> 0.85
```

两步之后就已经在 1.0 附近，三步之后 0.85。**没有任何一步依赖新的手写 kernel** ——
之前以为必须要的"原生 int8 direct conv"，在有了折叠 Q8（E19）之后不再是必需品，
它现在只是"把 0.85 再推到 0.6"的可选加速项。


---

## E25 TTFA 测不出来：设备已经不是"可测状态"了

想用 P2a 替身量"LM 快了之后 TTFA 是多少"。结果：

| 配置 | prompt | generation（16 帧） | TTFA |
|---|---:|---:|---:|
| chunk 8, r1 | 0.32 s | 5.38 s | **3.47 s** |
| chunk 8, r2 | 0.22 s | 8.37 s | 7.30 s |
| chunk 4, r1 | 0.13 s | 14.94 s | 13.02 s |
| chunk 4, r2 | 12.25 s | 13.95 s | 24.08 s |

同一份命令，TTFA 从 3.5 s 到 24 s。generation 336–933 ms/帧 —— 而 E23 里同样的替身只要 **44 ms/帧**。
差 8–20 倍，只能是机器状态。

查了一下：

```
/proc/loadavg: 19.85 16.05 14.78     （8 核）
ps -A | grep llama: 1 个残留 llama-tts（RSS 5.6 GB）+ llama-paraformer-api 服务
thermal zones: 38–42 °C
```

**有一个被 kill 掉的任务留下的 `llama-tts` 还在跑**（RSS 5.6 GB），加上这台设备本身还在跑别人的
paraformer 服务。我最近几轮的部分数据（尤其是 TTFA 和小 chunk 的）都被它污染了。

杀掉之后 load 仍是 19 —— 说明还有别的负载，设备已经不适合做延迟测量。

### 结论与协议

1. **TTFA 在当前设备状态下的观测值（3.5–24 s）不可用**，不能作为结论。
   唯一可信的是 E21/E23 那种"同一进程内稳态"的数字。
2. **做 TTFA 测量前必须**：`kill` 掉所有残留 llama 进程 -> `/proc/loadavg` < 2 -> 预热 15 s -> 再测。
   这一条要写进任何后续实验的开头，否则数据没有意义。
3. **按已测的稳态分量推算**，TTFA 的理论值是
   `0.1（prefill）+ 4×44（P2a 后 LM）+ (500 + 4×111)（首个 vocoder chunk）≈ 1.2 s`，
   但**这个数没有在干净设备上验证过**，只能标注为"推算"。

### 本轮的实际产出

- vocoder 成本归属（E24）：残差 72% / 非残差 28%，两平台一致 —— **这条是干净的**；
- 设备 trim 的预算锚点：111 -> 84 -> 68 ms/帧（后两个是 MAC 外推）；
- TTFA 的测量协议（上面第 2 条），以及"设备不可测"这个事实本身。

**剩下的工作已经不是"再试一个优化"，而是把 25 个实验收敛成一份可交付的结论。**
下一轮不再做新实验，直接产出最终方案文档（现状、可行路径、每一步的实测依据、复现方法、以及"v73 做不到的那部分该换什么硬件"）。


---

## E26 TTFA 的真凶找到了：**每个图第一次执行要付 6–10 ms/节点**，共约 18 s

带着 trace 看 TTFA 那一段（P2a 替身、chunk=8、设备）：

```
encode compute= 5 810.7 ms  (nodes=559)   <- vocoder 图，第一次执行
chunk = 8 frames
encode compute=12 205.9 ms  (nodes=2010)  <- predictor 图，第一次执行
gen_code 12 207.8 ms                       <- 就是这 12 秒
talker decode 105.1 ms
encode compute=    12.9 ms  (nodes=2010)   <- 之后每次 13–16 ms
gen_code 14.3 ms
talker decode 31.4 ms
...
```

- **第一次执行：2010 节点要 12.2 s ≈ 6 ms/节点**；同一张图之后只要 14 ms（7 µs/节点）。
- 差 **1000 倍**，而且 `majflt=0`、`minflt` 只涨 4 —— **不是缺页，不是加载，不是频率**（全程 2304 MHz）。
- 耗时与**节点数**近似成正比（vocoder 559 节点 5.8 s，predictor 2010 节点 12.2 s），与权重字节数不成比例。

所以 TTFA 的结构是：

```
TTFA = 首次图初始化（5.8 s vocoder + 12.2 s predictor ≈ 18 s）+ 稳态首包（~1.2 s）
```

### 这解释了之前所有的"随机慢"

- 之前只预热了 vocoder（一次 encode），**没有预热 predictor 图** → 第一个真实请求永远要付那 12 s；
- E11/E21 的"首 chunk 3–99 s"里，大头就是这个 per-graph 首次初始化（叠加爬频）；
- 我上一轮把 TTFA 归因为 DVFS，那个结论只对了一半：**DVFS 解释过渡区，首次执行开销解释 10 s 级的大数**。

### 产品含义

1. **预热必须是"完整跑一帧"**（GEN_CODE 一帧 + GEN_WAV 一个 chunk），启动成本 ~18 s；
   之后每个请求的 TTFA 才是 ~1.2 s。
2. 冷启动不能流式 —— 第一条请求要么等 18 s，要么在 App 启动时就做掉。
3. 这个开销与节点数成正比，所以 **P2a（4165 -> 1355 节点）还能把它从 12.2 s 降到 ~4 s**，
   这是 P2a 的一个额外收益（之前没算到）。


---

## E27 量化档位的取舍曲线：Q8_0 严格优于 Q6_K / Q5_K_M

上一轮说"如果 22.8 dB 不能接受，可退到 Q6_K"。直接量了这条曲线（折叠权重，host，16 帧）：

| 变体 | ms/帧 | 相对 F16 | 与 F16 的 SNR |
|---|---:|---:|---:|
| 原版 F16（未折叠） | 65.0 | 1.00 | — |
| **折叠 Q8_0** | **44.1** | **0.68** | **22.8 dB** |
| 折叠 Q6_K | 45.3 | 0.70 | **16.7 dB** |
| 折叠 Q5_K_M | 72.9 | 1.12（更慢） | 15.1 dB |

两点结论：

1. **Q6_K 与 Q8_0 一样快，但 SNR 从 22.8 掉到 16.7 —— 被 Q8_0 严格支配**，没有理由选它。
2. **Q5_K_M 反而更慢**（K-quant 的逐块反量化在 CPU 上比 Q8_0 的向量化点积更贵），
   同时又更差 —— 也是被支配的。

所以"退到 Q6_K"这个备选方案**不存在**：vocoder 的量化点只有两个，F16（65 ms，无损）或 Q8_0（44 ms，22.8 dB）。
取舍只能在这两者之间做，而判断依据只能是盲听。

**顺带**：这条曲线也说明 vocoder 的 DAC 卷积对位宽非常敏感（8.5 bit -> 6.6 bit 就掉 6 dB），
所以 P2b 的"减重重训"很可能也必须在 Q8_0 上做，不能再压位宽。

