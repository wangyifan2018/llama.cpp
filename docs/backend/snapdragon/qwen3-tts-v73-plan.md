# Qwen3-TTS-0.6B 在 v73（SM6650）上的最终优化方案

> 本文是 12 轮、27 个实测实验（[qwen3-tts-opt-log.md](qwen3-tts-opt-log.md)）的收敛结论。
> 所有数字都标注了出处（E 编号），未标注的是外推。

---

## 0. 一页结论

| 问题 | 答案 |
|---|---|
| v73 今天能跑多快 | **RTF 3.74**，TTFA 7.2 s（chunk 72） |
| RTF<1 可行吗 | **可行，但要三件套**：P2a 重训 + vocoder 减重重训 + 通道收窄 → **0.85** |
| 哪一步最关键 | **P2a**：LM 182 → 44 ms/帧（实测 -76%），是一步跨过 1.39 的唯一手段 |
| TTFA 能做到多少 | 预热做满后约 **1.2 s**（推算，未在干净设备验证） |
| 瓶颈现在在哪 | **vocoder**：111 ms/帧，占 wall 的 100%（P2a 之后） |
| 需要多少工期 | 8–12 周（3 项重训），外加 1–2 周后端/工程 |
| 有捷径吗 | 没有。折叠 int8（已做）之后，"手写 direct conv" 从必需降级为可选 |
| 想要稳的 RTF<0.5 | **换 v79（HMX）**，同样的活占算力 0.5% |

---

## 1. 今天的实测状态（全部在设备上、热态、预热后）

| 段 | 每帧 | 出处 | 备注 |
|---|---:|---|---|
| talker decode（28 层，CPU） | **14 ms** | E15 | HTP 上 52–66 ms，更慢 |
| code_predictor（16 次前向） | **168–197 ms** | E11/E13 | 节点 4165 |
| 采样（155 k 词表） | **0.1–0.75 ms** | 第 11 轮插桩 | 不是瓶颈 |
| **LM 合计** | **182 ms** | E20 | |
| **code2wav** | **98–111 ms** | E21/E24 | 折叠 Q8；残差单元占 72% |
| **端到端** | **295 ms（串行）** | E20 | **RTF 3.74** |

### 已经落地的优化（都在本仓库里，可直接复现）

| # | 改动 | 效果 | 出处 |
|---|---|---|---|
| 1 | code2wav 按实际帧数建图（不再固定 72） | vocoder -72%（24 帧），bit-exact | E02 |
| 2 | chunk 1/2/4 可调（`TTS_CHUNK`） | TTFA -66% | E03 |
| 3 | predictor 去掉 KV 搬移与 mask（增长前缀） | 节点 6460 -> 4724，bit-exact | E13 |
| 4 | DAC 卷积折叠 k 维 + int8 | vocoder -21~25%，22.8 dB | E19 |
| 5 | 预热协议（完整跑一帧，~18 s 启动成本） | 首请求 7 s -> ~1.2 s | E21/E26 |

代码入口：`tools/tts/tts.cpp`、`tools/mtmd/models/qwen3tts-gen.cpp`、`tools/mtmd/mtmd-*`、`conversion/qwen3tts.py`，
以及测量工具 `tools/tts/vocbench.cpp`。

---

## 2. 到 RTF<1 的三步

| 步骤 | 内容 | 收益 | 依据 | 工期 | 风险 |
|---|---|---|---|---|---|
| **P2a** | predictor 从 16 次前向改成 1 次（15 个并行头） | LM 182 -> **44 ms** | **设备实测**（结构替身 E23） | 3–4 周 | 质量（codes 一致性） |
| **P2b-1** | DAC 残差单元 3 -> 2 | vocoder 111 -> **84 ms** | 72% 归属实测 (E24) + host trim 实测 (E18) | 3–4 周 | 音质（3.1 dB 的坑靠重训补） |
| **P2b-2** | 通道 768/384/192/96 -> 640/320/160/80 | 84 -> **68 ms** | MAC C² 外推 | 2–3 周 | 音质 |
| 合计 | | wall = max(44, 68) = 68 | | **8–12 周** | **RTF 0.85** |

**只做 P2a：1.39。加上 P2b-1：1.05。三步齐：0.85。** 少一步都不行。

### 为什么 P2a 是唯一值得先做的

- 它是**收益最大且已实测**的一步（-76% LM，节点 -67%）；
- 它同时是 **HTP 的前置条件**：4165 节点的图在 v73 的 ggml-hexagon 上直接崩（`dspqueue_read 0x2e`，E15），1355 节点才有机会；
- 它不改音质以外的任何东西（结构、接口、调度都不变）。

规格已单独成文：[qwen3-tts-predictor-redesign.md](qwen3-tts-predictor-redesign.md)。

---

## 3. TTFA

```
TTFA = prefill + N×LM + (vocoder 固定开销 + N×111 ms)
```

| 配置 | N=4 | N=8 |
|---|---:|---:|
| 今天 | 1.8 s | 2.9 s |
| P2a 后 | **1.2 s** | 1.7 s |
| P2a + 减重 | 1.1 s | 1.5 s |

**前提：启动时必须完整预热一次"一帧"（GEN_CODE + GEN_WAV），成本约 18 s。**

这不是可选的：**每个图第一次执行要付 6–10 ms/节点**（E26），
vocoder 图 559 节点 5.8 s、predictor 图 2010 节点 12.2 s；之后同一张图只要 14 ms。
冷进程的第一条请求必然要等这 18 s，所以它必须发生在 App 启动时，不能发生在第一个用户请求上。

预热做满之后，上表的三行才成立（稳态首包由 prefill + N 帧 LM + 首个 chunk 组成）。
⚠️ 这三行仍是**推算**：设备在测量时被残留进程/他人服务占满（loadavg 19，E25），
干净验证需要在 "kill 残留进程 -> loadavg<2 -> 完整预热 -> 再测" 之后做。

**额外收益**：这个首次开销与节点数成正比，所以 P2a 会让它从 12.2 s 降到 ~4 s —— 之前没算进 P2a 的收益里。

---

## 4. 复现方法（从零开始）

```sh
# 1) 模型（modelscope 可直连，hf-mirror 有 429 限流）
curl -L "https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base/resolve/master/model.safetensors" -o model.safetensors
curl -L ".../speech_tokenizer/model.safetensors" -o speech_tokenizer/model.safetensors
PYTHONPATH=/tmp/pylibs python3 convert_hf_to_gguf.py <dir> --outtype q8_0 --outfile backbone.gguf
QWEN3TTS_WAV_CONV_QUANT=q8_0 PYTHONPATH=/tmp/pylibs python3 convert_hf_to_gguf.py <dir> --mmproj --outtype q8_0 --outfile mmproj-fold.gguf

# 2) host 构建（纯 CPU，Metal 在这台 M2 上慢 200x）
cmake -B build-host-cpu -S . -DGGML_METAL=OFF -DGGML_BLAS=OFF && cmake --build build-host-cpu --target llama-tts llama-vocbench -j 6

# 3) 设备构建（HTP 在 docker 镜像里，不在本机 NDK）
docker run --rm -v "$PWD":/workspace -w /workspace -u $(id -u) \
  ghcr.io/snapdragon-toolchain/arm64-android:v0.8 \
  bash -c "cmake --build /workspace/build-adb --target llama-tts llama-vocbench -j 8"

# 4) 测量（设备上）
adb push backbone.gguf mmproj-fold.gguf ref.wav /data/local/tmp/q3tts/
adb shell 'cd /data/local/tmp/q3tts && pkill -9 -f llama-tts; cat /proc/loadavg'   # 必须 < 2
TTS_TRACE=1 TTS_STREAM=1 TTS_CHUNK=4 ./llama-tts ... -dev none -n 24   # 预热 2 次后再测
./llama-vocbench -dev CPU|HTP0 -n 32 -c 2    # 单测 vocoder
```

测量仪器（都是环境变量开关，默认关闭）：
`TTS_STREAM`（TTFA/RTF）、`TTS_CHUNK`（分块）、`TTS_TRACE`（每段 ms）、`TTS_TRACE_OPS`（算子直方图）、
`TTS_FAKE_NAR`（1-pass 结构替身）、`TTS_VOC_RES`（残差单元数）。

---

## 5. 做不到的部分（诚实边界）

| 项 | 状态 | 原因 |
|---|---|---|
| 端到端 RTF<1 | **未实测**，只到 0.85 的推算 | 需要 8–12 周重训 |
| TTFA < 500 ms | 不可能 | vocoder 固定开销（~500 ms/调用）+ 稳态 111 ms/帧 |
| HTP 跑 LM | 不可行 | v73 无 HMX，GEMV 上 HTP 慢 3.7x |
| HTP 跑 vocoder | 无收益 | conv 被降级成 im2col+GEMM，与 CPU 持平（E16） |
| int8 音质 | 22.8 dB，**未过 30 dB 验收线** | 需盲听裁决；**没有 Q6_K 退路**（E27：Q6_K 同速但只有 16.7 dB，被 Q8_0 严格支配） |
| predictor 质量 | 完全未验证 | 替身用的是 AR 权重 |

### 换硬件的判断

v79（HMX，12 TOPS fp16，8 MB VTCM）：
- vocoder 31 GMAC/s 折 fp16 = 62 GFLOP/s，占 0.5%；
- LM（P2a 后）0.53 GMAC/帧，占 0.02%；
- HTP 的 conv/GEMV 效率也不再是瓶颈。

**结论：同一个模型，换一代 SoC，RTF 从"三件套勉强 0.85"变成"随手 0.2"。**
如果 RTF<0.5 或 TTFA<200 ms 是硬指标，应直接换机型，而不是继续在 v73 上投重训。

---

## 6. 下一步（按优先级，都不需要新硬件）

1. **P2a 重训**（唯一已实测的大杠杆，-76% LM）。判据：codes top-1 一致率 >= 60% + 盲听。
2. **vocoder 减重重训**（-24.5%）。
3. **TTFA 在干净设备上验证**（1 小时的活，只要设备空闲）。
4. 可选：direct int8 conv（把 0.85 推到 0.6）。~~Q6_K 折中~~ 已排除：同速且更差（E27）。

### 量化档位的唯一取舍（E27）

| 档位 | vocoder | SNR | 结论 |
|---|---:|---:|---|
| F16（不折叠） | 65 ms/帧 | 无损 | 安全但慢 |
| **Q8_0（折叠）** | **44 ms/帧** | 22.8 dB | **唯一可用的加速档**，需盲听 |
| Q6_K / Q5_K_M | 45 / 73 ms | 16.7 / 15.1 dB | 被 Q8_0 支配，排除 |
