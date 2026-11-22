# Qwen3-TTS 走 QNN 的优化方案（面向 v81）

> 2026-11-22。前提：v81 上可能不用 llama.cpp 的 ggml-hexagon，而用 QNN（高通 AI Engine Direct）。
> 本文回答：换了 QNN 之后哪些结论仍然成立、哪些作废、哪些从可选变成必须，以及现在就该做的准备。

---

> **2026-11-22 修正**：本文最初写的「QNN 没有 Sin、没有 TransposeConv」是错的。 按 ORT QNN EP 官方支持算子表，
> Sin 与 ConvTranspose 都支持；真正的约束是 **MatMul 在 HTP 上只接受量化类型**（fp16 会回退 CPU）。
> 详见 qwen3-tts-qnn-op-matrix.md。

## 0. 一句话

**QNN 把「算子能不能上 NPU」变成「图能不能编译」，评判标准从算子覆盖率变成图调用次数与布局。**
对我们影响最大的一条：**predictor 的 16 次串行前向在 QNN 下会变成 16 次图调用**，
每次都带几十到几百微秒固定开销 —— 所以 **P2a（1-pass predictor）在 v81+QNN 上从可选优化变成必做项**，
优先级高于 vocoder。

---

## 1. QNN 与 ggml-hexagon 的差别（决定优化取向）

| 维度 | ggml-hexagon（现在） | QNN |
|---|---|---|
| 执行模型 | 逐算子调度，100-137 us/op（E15 实测） | 整图离线编译成 context binary，运行时一次 FastRPC |
| 融合 | 靠 htp-fusion 少量模式 | 编译器自动融合（conv+act、matmul+bias） |
| 算子覆盖 | 实测 91% 节点支持、9% 回退（E41） | 覆盖窄但深：Conv2d / TransposeConv2d / MatMul / ElementWise |
| 缺失算子 | Sin 无（319 次回退）、Col2im 无 | 仍无 Sin（M5 评估确认），但有 TransposeConv |
| 布局 | ggml 的 ne0 最快 | NHWC / CRD 偏好，需要重排 |
| 形状 | 动态，每次 encode 重建 | 静态，要按 bucket 定形状 |
| 精度 | fp16/fp32/int8 自定 | QNN 自己的量化编码（w8a8 / w8a16） |
| 调用开销 | 每 op 一次提交 | 每图一次，所以「少调用、每次多干活」是核心策略 |

**推论**：QNN 下算子数不再重要，「图调用次数 x 每次图的工作量」才重要。

---

## 2. 三条已有结论在 QNN 下的命运

| 结论 | QNN 下 |
|---|---|
| vocoder 111 ms/帧，残差单元占 72%（E24） | 仍成立（模型性质）。QNN 能融合 conv+Snake，但 v73 无 HMX 仍无大收益；v81 有 HMX 才是 3-5x 的地方 |
| predictor 16 次串行 168 ms（E11/E13） | **在 QNN 下更糟**：16 次图调用 + 每次固定开销 |
| 投机解码 1.07-1.18x（E37） | 可能变好：把 15 次调用压成 6-8 次，而 QNN 的固定开销占比比 ggml 更高 |
| 折叠卷积（K*IC 矩阵，E19） | **在 QNN 下应放弃**：QNN 有原生 Conv2d，折叠会破坏布局与自动融合 |
| int8 只省 3.6%（E19） | 不适用：QNN 用自己的量化，需要单独标定 |

---

## 3. v81+QNN 的优先级（与 v73 相反）

    v73/ggml : vocoder(111) 是瓶颈 -> 先压 vocoder
    v81/QNN  : vocoder 可能被 HMX 压到 30-50 ms
               -> 瓶颈变成 predictor 的 16 次串行调用
               -> P2a（1-pass）从可选变必做

| 顺序 | 项 | 为什么在 QNN 下更值 | 需要重训 |
|---|---|---|---|
| 1 | **P2a：1-pass predictor** | 16 次图调用 -> 1 次；QNN 固定开销被彻底摊掉 | 是（3-4 周） |
| 2 | vocoder 上 QNN（原生 Conv2d + TransposeConv + HMX） | 编译融合 + HMX；顺带消掉 Sin 回退 | 否 |
| 3 | Snake 多项式化 | QNN 没有 Sin 算子，不换就编译不过或回退 CPU | 否 |
| 4 | 投机解码（在 QNN 下重估） | 减少调用次数的另一条路，与 P2a 二选一 | 否 |
| 5 | 并行化 | 仍有效，但 QNN 下两者可能都在 DSP，抢同一颗 | 否 |

---

## 4. 现在（v73 上）就该做的三件 QNN-ready 准备

共同点：在 v73 上就能验证，且是 QNN 迁移的前置条件。

### 4.1 Snake 去 Sin（最高优先）

M5 的评估已经量化过：Sin 占 codec kernel 时间 14%，而且 HVX/QNN 都没有 Sin 算子
（M5 的解法是 9 阶多项式 + 范围归约，见 m5_snake_op.h，误差 1e-4，已上生产）。

我们的 vocoder 里有 **319 个 SIN 节点**（E41），在 HTP 上全部回退 CPU。换成多项式后：

- HTP 回退从 1384 降到约 1065（去掉 319）；
- CPU 路径也更快（无 libm sin）；
- 这是 QNN 能编译通过的前提。

数学形式（Snake）：y = x + sin^2(a*x)/b，其中 a=exp(alpha)、b=exp(beta)。
sin^2 是偶函数，先折到 [-pi/2, pi/2]：t = t - round(t/pi)*pi，再用偶次多项式：

    sin^2(t) ~= t^2 - t^4/3 + 2*t^6/45 - t^8/315      (|t| <= pi/2, 误差 < 1e-4)

ggml 里需要的算子都有：ggml_round、ggml_sub/mul/div、ggml_sqr。**纯图改造，不改后端。**

**验收**：与现路径逐样本 SNR >= 60 dB（1e-4 误差约对应 80 dB），再跑 llama-vocbench 看 ms/帧。

### 4.2 转置卷积保持原生形态

我们在 ggml 路径上把 CT 折成了 mul_mat（E19）。QNN 有原生 TransposeConv2d，
所以 ONNX 导出时应保持原生 CT，不要带折叠。这意味着要维护两套图结构
（ggml 折叠版 / ONNX 原生版），建议在导出脚本里用开关分叉，而不是改模型定义。

### 4.3 导出管线：ONNX

| 子图 | 形态 | 备注 |
|---|---|---|
| talker | 单步 decode（KV 作为显式 I/O） | 与 v73 的 ggml 图等价即可 |
| predictor | 1-pass（P2a 之后）；过渡期可 16 步展开图 | 静态形状：16 位置固定 |
| vocoder | 一个 chunk（如 8 帧）一张图，原生 Conv/TransposeConv/Snake | 输出 8x1920 = 15360 样本 |

导出用 torch.onnx.export 从参考实现（qwen_tts 包）导出，**不要**从 GGUF 反向重建
（GGUF 已经折叠/量化过）。

---

## 5. v81+QNN 的第一批实验（按顺序）

| # | 实验 | 判据 | 依赖 |
|---|---|---|---|
| 1 | 在 v81 上跑 qnn-net-run 样例，确认 HTP + HMX 可用 | 能跑通且 htp_performance_mode=burst 生效 | QNN SDK + v81 设备 |
| 2 | vocoder 单 chunk 的 QNN context binary vs CPU | >= 2x 才算成功 | ONNX 导出 + converter |
| 3 | predictor 1-pass 的 QNN 版本 vs 16 步版本 | 调用次数 16->1 的端到端收益 | P2a 权重或替身 |
| 4 | 量化：QNN w8a8 标定后的 SNR / 盲听 | SNR >= 20 dB（对齐 E19 的 22.8 dB） | 标定集 |
| 5 | 端到端 RTF / TTFA | RTF < 1、TTFA < 1.5 s | 1-4 全通 |

**注意**：第 1 步之前拿不到 QNN SDK 就全部做不了。M5 的评估已确认：
设备侧运行时齐全（/odm/lib64/libQnnHtp*.so），缺的是构建侧 SDK（开发者站下载）。

---

## 6. 风险

1. **静态形状与自回归冲突**：每帧一次图调用不可避免；若一帧要 3 张图（talker/predictor/vocoder），
   每秒 200 帧就是 600 次 FastRPC。**这是 QNN 路线最大的结构性风险**，也是 P2a 必须做的原因。
2. **量化不可控**：QNN 的 w8a8 由它自己插 Q/DQ，无法像 GGUF 那样逐张量调；
   vocoder 的 Snake 对量化敏感（E27：8.5->6.6 bit 掉 6 dB）。
3. **工程量大**：两条图路径（ggml 与 ONNX）长期并存，回归成本翻倍。
4. **v73 无法验证收益**：v73 无 HMX，QNN 在 v73 上大概率与 CPU 打平（M5 已验证 HTP 小算子更慢）。

---

## 7. 建议的决策

| 情形 | 建议 |
|---|---|
| 只做 v73 | 不上 QNN。继续走 ggml-hexagon/CPU，把 4.1（Snake）做完就好 |
| 确定要上 v81 | 现在做 4.1 + 4.3，**并立刻立项 P2a** —— QNN 下它的收益从 1.4x 变成「决定能不能落地」 |
| 不确定 | 先做 4.1（两边都赚，且是 QNN 前提），P2a 等 v81 决策 |

---

## 8. 与现有文档的关系

- qwen3-tts-v81-feasibility.md：ggml-hexagon 路线下的 v81 重估（91% 节点已在 HTP、9% 回退）
- m5-vocoder-npu-eval.md：同一颗芯片上的 NPU 评估（小算子慢、只有大 int8 GEMM 赚、无 HMX）
- qwen3-tts-speculative-decoding.md：投机解码（QNN 下需要重估）
- qwen3-tts-training-plan.md：P2a 训练计划（QNN 路线上它的优先级上升）

---

## 9. 怎么拿到 QNN 工具链（2026-11-22 实测可下载）

### 9.1 直接下载（不需要登录，推荐）

Qualcomm Software Center 的社区版 QAIRT（含 AI Engine Direct / QNN）：

    export QAIRT_VERSION=2.42.0.251225
    wget https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QAIRT_VERSION}/v${QAIRT_VERSION}.zip
    unzip v${QAIRT_VERSION}.zip
    cd qairt/${QAIRT_VERSION}
    source bin/envsetup.sh

实测：该 URL 返回 301 跳到 apigwx-aws.qualcomm.com，range 请求返回 **206**，
文件大小 **1,543,955,191 字节（约 1.44 GB）**，无需账号。2.40.0.251030 也可以同样下载。

环境要求（官方）：x86-64 Linux（Ubuntu 22.04）+ Python 3.10。
解压后有我们要的两类东西：

| 用途 | 路径 |
|---|---|
| 模型转换（ONNX -> QNN） | bin/x86_64-linux-clang/qairt-converter、qnn-onnx-converter |
| 设备侧运行时 | lib/ 下的 libQnnHtp.so、libQnnHtpV73Stub.so、skel/libs |

### 9.2 其它渠道

| 渠道 | 说明 |
|---|---|
| QPM（qpm.qualcomm.com） | 高通官方包管理器，需要账号；qpm-cli 安装同一个 SDK |
| Qualcomm AI Hub | 云端编译，不给本地 SDK；适合只要 context binary 的场景 |
| pip qai_appbuilder | 只有设备侧运行时封装（Python/C++ API），**不含转换器** |
| pip onnxruntime-qnn | Windows 用，1.18 起自带 QNN 库，不需要单独装 SDK；**Android 侧仍需 SDK 的转换器** |

### 9.3 对我们的意义

- 设备侧运行时**本来就有**（/odm/lib64/libQnnHtp*.so），所以我们要的是**转换器**，
  也就是必须在 x86-64 Linux 上跑 QAIRT。
- 本机是 macOS，跑不了 qairt-converter —— 需要一台 x86 Ubuntu（或 CI 上跑）。
- 版本对齐：ORT 的 QNN EP 官方测的是 QNN 2.22.x；QAIRT 2.42 更新很多，
  若走 ORT 路线要先确认 EP 版本与 SDK 版本匹配；走 QNN C API 直调则没有这个约束。