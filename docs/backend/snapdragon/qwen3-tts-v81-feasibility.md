# 迁移到 v81 的可行性（Qwen3-TTS 优化线）

> 2026-11-22。目标机从 v73(SM6650) 换成 v81 时的重估：哪些结论会变、哪些不变、先测什么。
> 事实来源分两类并标明：【仓库】= 本 checkout 的代码；【待测】= 需要在 v81 设备上实测，不能推断。

---

## 0. 一句话

**v81 的收益点很明确：HMX 加速 mul_mat，而折叠之后的 vocoder 有 91% 的节点是 mul_mat 且已经在 HTP 上跑。**
实测剩下 9% 回退 CPU 的是 SIN（DAC 的 Snake 激活，319 次）和 COL2IM_1D（转置卷积，66 次），
它们恰好落在"占 72% 时间"的残差单元与上采样里。所以 v81 的实际上限取决于能不能把这两类也算进 HTP。

反过来，**talker / predictor 不会因为 HMX 变快**：它们是单 token GEMV，瓶颈是带宽不是算力。
"LM 放 CPU、vocoder 放 HTP" 这个分工在 v81 上大概率仍然成立。

---

## 1. 仓库里已经有的 v81 通路（不需要改代码）

| 事实 | 位置 |
|---|---|
| 工具链**无条件**给 DSP 编 `-mhmx` | `htp/cmake-toolchain.cmake:98`：`ARCH_FLAGS "-mcpu=${V_ARCH} -m${V_ARCH} -mhvx=${V_ARCH} -mhmx"` |
| 运行时按 hwinfo 决定用不用：`n_hmx = opt_nhmx ? hw_n_hmx : 0` | `ggml-hexagon.cpp:2293-2315` |
| 环境变量开关 `GGML_HEXAGON_NHMX=0/1` | `ggml-hexagon.cpp:3598, 3666` |
| conv2d 有 HMX 路径：`HTP_CONV2D_KERNEL_HMX_WINDOW` / `HMX_TILED`（tiled implicit-GEMM） | `htp/conv2d-ops.h:14-15`、`conv2d-ops.c:2420` |
| flash attention 有 HMX tile 常量（32x32 fp16） | `htp/flash-attn-ops.h:16-21` |
| 启动打印一行 hwinfo | `ggml-hexagon.cpp:2307` |

**移植第一步不是改代码，是看那一行日志**：

```
ggml-hex: <dev> hwinfo: threads N, hvx N, hmx N, vtcm N MB
```

v73 上是 `hmx 0, vtcm 2 MB`。v81 若 `hmx 1` 且 vtcm 更大，第 2 节的判断要重做；
若 `hmx 0`（CDSP 被裁剪），那 v81 只是 "更快的 v73"。

---

## 2. 哪些已有结论会变，哪些不变

| 结论（v73 实测） | v81 上会怎样 | 理由 |
|---|---|---|
| **vocoder 111 ms/帧，残差单元占 72%** | **大概率大幅下降** | v73 的 conv 被降级成 im2col+GEMM（E16 实测"与 CPU 持平"正因如此）；HMX_TILED 是原生隐式 GEMM |
| **vocoder 在 HTP 上 ≈ CPU** | **会翻转** | 同上 |
| **talker 在 HTP 上慢 3.7x（52-66 vs 14 ms）** | **不变** | 单 token GEMV 是带宽问题；HMX 只加速矩阵块乘 |
| **predictor 4165 节点在 HTP 上崩（dspqueue 0x2e）** | 可能仍崩 | 与 HMX 无关，是图规模/内存问题；1-pass（1355 节点）才有机会 |
| **每行验证成本 ≈ 2.6 ms（推算）** | **下降** | 验证是"多行一次前向"，行数越多越像 GEMM |
| **投机解码 1.07-1.18x（E37）** | **可能变好** | 盈亏条件 `T2/T1 < 1+a`；HMX 拉低 T2/T1，a 不变 |
| **int8 只省 3.6%（被 F16/F32 规则挡住）** | 不变 | 转换器规则问题，已用折叠方案解决 |
| **Q6_K 被 Q8_0 支配** | 不变 | 模型性质 |

**结论**：v81 改变的**只有 vocoder 与验证型算子**，不改变 talker/predictor 的分工与量化结论。

---

---

## 2.5 实测修正：91% 的 vocoder 算子已经在 HTP 上，回退的是 Snake 与转置卷积

早先我从源码推"conv1d 不在 HTP 支持列表，所以 vocoder 没上 HTP"——**实测证否**。
折叠权重之后卷积走 concat + mul_mat，而 mul_mat 在 HTP 上受支持。
用 GGML_HEXAGON_VERBOSE=1 统计折叠 Q8 的 vocbench（-dev HTP0 -n 4）：

    supported 13050, rejected 1384        （按节点数，HTP 吃下 91%）

真正回退 CPU 的算子：

| 算子 | 次数 | 在哪里 |
|---|---:|---|
| ARANGE | 594 | 通用索引构造 |
| MUL_MAT | 347 | 形状/缓冲不合规的那部分（另有 2056 个 MUL_MAT 是支持的） |
| **SIN** | **319** | **DAC 的 Snake 激活** |
| **COL2IM_1D** | **66** | **转置卷积（上采样与 DAC 输出）** |
| PAD_REFLECT_1D | 44 | 反射填充 |
| MEAN | 14 | — |

### 对 v81 判断的影响

1. **v81 的 HMX 会有收益**：它加速 mul_mat，而 mul_mat 占 91% 的节点。
   前提是那 9% 的 SIN / COL2IM 别把流水线卡住（每次回退都要跨设备同步）。
2. v73 上"vocoder 在 HTP 无收益"的原因**不是**算子映射，而是 HVX 上这些 GEMV 带宽受限，与 CPU 同量级。
3. 因此 v81 的收益上限，取决于能不能把 SIN 和 COL2IM 也留在 HTP。

### 两个前置优化项（v73 上就能做，不需要重训）

| 项 | 做法 | 风险 |
|---|---|---|
| 消掉 SIN 回退 | Snake 的 sin 没有 HTP 算子。可以批量做（一次同步处理所有 Snake）、用多项式逼近（误差 <1e-3）、或把它融进相邻卷积 | 逼近会改变输出，需要 SNR 对比 |
| 消掉 COL2IM_1D 回退 | 转置卷积改写成「显式上采样 + 普通卷积」（零插值 + conv），完全避开 col2im | 计算量增加 K*stride 倍，要实测权衡 |

这两项是"让 HTP 真正吃满 vocoder"以及"v81 迁移能兑现"的共同前置条件。

## 3. 到 v81 之后方案排序会翻转

```
v73：Q8 predictor(97.7) + vocoder(111) 串行 -> 并行 -> vocoder 减重 -> 0.85
v81：Q8 predictor(97.7, CPU) + vocoder(?, HTP+HMX) -> 若 vocoder 掉到 30-50 ms
     则 wall = max(97.7, 40) = 97.7 -> RTF ~1.2，predictor 变成新瓶颈
```

| 顺序 | v81 上的优先项 | 为什么 |
|---|---|---|
| 1 | **vocoder 上 HTP + HMX**（先看 hwinfo，再跑 `llama-vocbench -dev HTP0`） | 可能一次拿掉 60-80 ms |
| 2 | **predictor 全主干 Q8 + 尾部学生**（E32/E33 路线） | 它成新瓶颈后，97.7 -> 60 是主要收益 |
| 3 | 投机解码**重新评估** | T2/T1 下降后可能从 1.07x 变 1.3x+ |
| 4 | 并行化 | 仍有效，但总量小了收益变小 |

---

## 4. 到 v81 后第一批测量（工具都是现成的）

| # | 命令 | 看什么 | 判据 |
|---|---|---|---|
| 1 | `llama-tts ... -dev none` 随便跑 | `ggml-hex: ... hwinfo: ... hmx N, vtcm N MB` | hmx=1 且 vtcm >= 4 MB 才值得做第 2 步 |
| 2 | `llama-vocbench -dev HTP0 -n 32 -c 2` vs `-dev CPU` | ms/帧 | HTP 明显快 = HMX 生效 |
| 3 | `GGML_HEXAGON_NHMX=0` 重跑 #2 | 差值 | 差值即 HMX 净贡献 |
| 4 | #2 与 `TTS_VOC_RES=0` 组合 | 残差隐含占比 | 若占比下降，减重的收益也要重估 |
| 5 | `llama-tts -dev none` 端到端 | RTF | 与 v73 的 2.677（E36）对比 |

`-dev HTP0` 需要 `LD_LIBRARY_PATH=./htp/lib`。

---

## 5. 风险与不要做的事

1. **不要假设 v81 一定有 HMX**。v73 上我们已经见过 `hmx 0`（CDSP 报不支持）。先看日志。
2. **不要假设 VTCM 更大**，让日志说话（v73 是 2 MB，tiled 方案就是围绕它设计的）。
3. **不要因为换硬件就重训**。E35/E36 已证明 Q8 predictor 几乎无损（Mac 重合 95.65%），这条在 v81 照样成立；
   重训优先级由"predictor 是否成为新瓶颈"决定，不由硬件代号决定。
4. **v81 的 CPU 核也变了**，`-t 4` 的最优线程数要重扫。

---

## 6. 本机查不到、只能上机量的信息

- v81 的 HMX 是否对第三方 CDSP 开放（不只给 Qualcomm 自家 SDK）。
- v81 的 VTCM 实际大小（v73=2 MB；v79/v81 的公开数字我没有可靠来源，**不猜**）。
- 新 CPU 的核配置与频率。

这三条都用第 4 节 #1 的日志和第 5 节命令量出来。
