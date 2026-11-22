# Qwen3-TTS 算子 vs QNN HTP 支持矩阵

> 2026-11-22。来源：ONNX Runtime QNN Execution Provider 官方支持算子表（onnxruntime.ai，QNN EP 文档）。
> 这份表针对 **ORT + QNN EP** 这条最省事的路线；直连 QNN SDK 的算子集更大，但先按最严格的来评估。

---

## 0. 结论先说：比预期好，**不需要自定义算子**

上一轮我在 QNN 方案里写了两条错的判断，这次用官方表纠正：

| 我上一轮写的 | 实际 |
|---|---|
| QNN 没有 Sin，Snake 要改多项式/自定义算子 | **错**：`ai.onnx:Sin` 在支持列表里 |
| QNN 没有 TransposeConv，转置卷积要拆 | **错**：`ai.onnx:ConvTranspose` 支持（3d 自 1.18） |

所以 vocoder 的三类关键算子（Conv / ConvTranspose / Sin）**QNN 全都有**，
E41 里那 9% 的 ggml 回退（SIN 319、COL2IM 66）在 QNN 上**不存在**。

**真正的约束换成了另一条**：`ai.onnx:MatMul` 在 HTP 上**只接受量化类型**
（uint8 x uint8 / uint8 x uint16 / uint16 x uint8）。
也就是说：**fp16 的 MatMul 会掉回 CPU**，想留在 HTP 就必须量化。

---

## 1. 我们用到的算子 vs QNN HTP

| 我们的算子（E41 dump / 代码） | 次数（vocoder 图） | QNN HTP | 备注 |
|---|---:|---|---|
| MUL_MAT | 2403 | `MatMul` / `Gemm` | **仅量化类型**，fp16 会回退 CPU |
| MUL / ADD / SUB / DIV | ~3500 | 支持 | — |
| RESHAPE / CPY | ~2200 | 图重组，不产生算子 | 导出时会被折叠 |
| RMS_NORM | 1053 | 分解为 Mul/Pow/ReduceMean/Sqrt | 都是支持的；或用 LayerNormalization 近似（注意定义不同） |
| CONCAT | 967 | `Concat` | — |
| SIN | 319 | **`Sin` 支持** | Snake 直接用 |
| COL2IM_1D | 66 | **`ConvTranspose` 支持** | 导出时保持原生转置卷积，不要拆 |
| PAD_REFLECT_1D | 44 | `Pad` | **reflect 模式要验**，列表里只写 Pad |
| MEAN | 14 | `ReduceMean` | — |
| ARANGE | 594 | `Range` **不在列表** | 位置索引是静态的，导出时折叠成常量 |
| GET_ROWS | — | `Gather` | **只支持非负索引**（我们满足） |
| SOFT_MAX / ARGMAX | — | `Softmax` / `ArgMax` | 采样可放图内 |
| LAYER_NORM | — | `LayerNormalization` | tagger/predictor 用到 |

---

## 2. 三条落地约束（按影响排序）

### 2.1 MatMul 必须量化

QNN HTP 的 MatMul 只吃 uint8/uint16。这意味着：

- predictor 与 talker 的权重必须走 **w8 量化**（我们已有 Q8 GGUF，但 QNN 要的是它自己的编码，需要重新标定）；
- vocoder 里的注意力 MatMul 如果保留 fp16，会掉回 CPU —— 而 E27 显示 vocoder 对位宽敏感（8.5→6.6 bit 掉 6 dB），
  所以 **w8a16 或 w8a8 的标定质量是这条路线成败的关键**。
- 折叠卷积（K*IC 矩阵）在 QNN 上**应当放弃**：直接用原生 Conv2d（QNN 会融合）。

### 2.2 静态形状

QNN 是编译期定形状。我们的三个子图都要按 bucket 固定：

| 子图 | 建议 bucket |
|---|---|
| talker | 单步 decode，KV 长度按 128/256/512 分档 |
| predictor | 1-pass（P2a）或 16 步展开；两者都是固定形状 |
| vocoder | chunk = 8 帧，输出 8x1920 样本 |

### 2.3 图调用次数

每帧 3 次图调用（talker/predictor/vocoder）x 12.5 帧/秒 = 37.5 次 FastRPC/秒，
可接受；但如果 predictor 保持 16 步（16 次/帧），就是 200 次/秒 —— **这是 P2a 在 QNN 上从可选变必做的量化依据**。

---

## 3. 还缺什么（阻塞项）

| 项 | 状态 |
|---|---|
| QNN SDK（构建侧） | **没有**。设备侧运行时齐全（/odm/lib64/libQnnHtp*.so），构建侧要开发者账号下载 |
| ORT 的 QNN EP | 设备上的 ORT 1.23.2 未编入（M5 评估实测 addQnn() 抛异常） |
| ONNX 导出 | 未做。需要从参考实现导出三个子图（本机 transformers 版本不够，见训练计划 §8） |
| 标定集 | 未准备。w8a8/w8a16 标定需要真实语音数据 |

**结论**：QNN 路线技术上比预期顺（算子齐全），但**被工具链挡在门外**：
没有 QNN SDK，且设备上的 ORT 不带 QNN EP。这两条不解决，一切实验都做不了。

---

## 4. 现在能做的准备（不需要 SDK）

1. **算子清单固化**：本文就是。以后每次改图，先对照这张表。
2. **ONNX 导出脚本骨架**：从参考实现导出 talker/predictor/vocoder，保持原生 Conv/ConvTranspose/Sin。
   导出后可以用 onnx 的 shape inference 做静态形状检查，不需要 QNN SDK 就能验证图是否合法。
3. **量化标定的数据准备**：与训练计划里的 shard 生成复用（同一批语音）。

---

## 5. 与其它文档的关系

- qwen3-tts-qnn-v81-plan.md：QNN 路线总方案（本文纠正其中关于 Sin/TransposeConv 的两条）
- qwen3-tts-v81-feasibility.md：ggml-hexagon 路线下的 v81 重估（91%/9% 的算子分布）
- m5-vocoder-npu-eval.md：同一芯片的 NPU 评估（HVX 侧经验，QNN 的结论不同）
