# SeACo-Paraformer 与说话人模型：可行性实测（2026-11-18）

## 0. 结论摘要

1. **SeACo 结构可行，但不建议现在换**：encoder 与主 decoder 与我们现用模型**逐层一致**，
   增量是 4 类模块（共约 27M 参数），需要新写 LSTM/BiLSTM 图与第二个 decoder，
   预计 +20~25 ms（约 +8% 的 endpoint->final），而收益（热词召回）与我们已有的
   `PF_CONTEXT` 前缀树重叠，且它只在**中文热词**上训练过。
2. **CAM++ 门控原型被环境卡住**：模型权重（27 MB）能下载，但本机装不上 `funasr`
   （pypi 走代理被重置，ModelScope 也没有 ONNX 导出），拿不到 CAMPPlus 的架构代码。
   我不想手写一份无法与参考实现对拍的 CAM++ 前向——那样得到的"可分性"数字不可信。

## 1. SeACo-Paraformer 的模块清单（从 model.pt 实测）

权重 `model.pt` = 989,763,045 字节（944 MB fp32），**247.3 M 参数**，1095 个张量。

| 模块 | 张量数 | 参数 | 与现有模型的关系 |
|---|---:|---:|---|
| `encoder.encoders` + `encoders0` | 650 | 161.2 M | **完全一致**（50 blocks，512/2048） |
| `decoder.decoders` + `output_layer` | 295 | 57.0 M | **完全一致**（16 blocks） |
| `seaco_decoder` | 115 | 14.8 M | **新增**：6 blocks，FFN 1024，cross-attn，kernel 21 |
| `bias_encoder`（2 层 LSTM） | 8 | 8.4 M | **新增**：weight_ih/hh (2048, 512) × 2 层 |
| `predictor.blstm` | 8 | 4.2 M | **新增**：CIF v3 的 upsample BiLSTM |
| `predictor.cif_conv1d` | 2 | 0.79 M | **新增**：CIF v3 的 conv1d |
| 其他（norm、bias 等） | 17 | ~0.9 M | — |

配置层面还有两个差异（`seaco_config.yaml`）：

- `predictor: CifPredictorV3`（我们实现的是 V2），`upsample_type: cnn_blstm`，
  `upsample_times: 3`；
- `model_conf: bias_encoder_type: lstm, bias_encoder_bid: false, inner_dim: 512, NO_BIAS: 8377`。

解码流程 = 主 decoder logits + seaco decoder logits（由热词 embedding 经 LSTM 得到 bias），
再按 `NO_BIAS` 掩码融合。

## 2. 移植代价

| 项 | 工作量 | 风险 |
|---|---|---|
| 转换器：新增 4 类张量、命名映射 | 1–2 天 | 低（照 keys 映射） |
| 运行时：2 层 LSTM（热词 ≤20 步，展开成图） | 2–3 天 | 中：每步 4 个 512×2048 matmul，展开后约 100 个 op |
| 运行时：CIF v3 的 BiLSTM upsampler（T=45 帧 × 2 方向） | 2–3 天 | 中：约 200 个 op，注意 HTP dispatch 成本 |
| 运行时：seaco decoder（可复用现有 decoder 构建函数，换权重与维度） | 2 天 | 低 |
| 融合 + 热词文本→token 管线（seg_dict 分词） | 2 天 | 中：分词语义要与 FunASR 对齐 |
| HTP 数值对拍 + 延迟验收 | 3–5 天 | 中 |

**预计延迟增量**：seaco decoder（6 blocks、FFN 1024）约 +15 ms；BiLSTM 约 +2–3 ms；
LSTM bias encoder + 融合约 +2 ms；合计 **+20~25 ms（约 +8%）**。
权重 Q8 后约 250 MB（现 227 MB），内存基本持平。

## 3. 是否值得换（对比我们已有的能力）

| 维度 | SeACo | 我们现状（`PF_CONTEXT` 前缀树） |
|---|---|---|
| 热词机制 | 训练得到的 LSTM bias | 解码期 logit 偏置（零成本） |
| 实测收益 | 论文在 AISHELL-hotword 上提升；本机无法复现（无 funasr） | 命名片段 6/10 -> 8/10，车控夹具零回归 |
| 语言 | 中文热词训练（AISHELL）；英文未知 | 中文精确、英文需显式 token |
| 延迟 | +8% | 0 |
| 工作 | 1–2 周 + 新模型代码 | 已完成 |

**建议**：不换。理由三条——(1) 它只在中文热词上训练，我们的英文场景不会受益；
(2) 拿到的是固定权重，无法用我们的 POI/联系人分布微调；(3) 我们要的"热词"本质上是
**运行时才知道的实体**，用解码期偏置更直接，且不赌模型训练数据。

**什么情况下重新考虑**：真实录音验证后，热词召回成为第一痛点，且有开发机可以跑 funasr
做 A/B（那时先用 Python 侧跑 SeACo 对比我们模型，确认增益后再决定是否移植）。

## 4. CAM++ 门控原型：当前阻塞与两条路

已完成：模型权重下载成功（`campplus_cn_common.bin`，28.0 MB，torch zip 格式），
配置确认（`feat_dim 80, embedding_size 192, growth_rate 32, bn_size 4, init_channels 128`）。

阻塞：本机 `pip install funasr` 失败（pypi 代理重置连接；ModelScope 无 wheel），
ModelScope 上也没有 CAM++ 的 ONNX 导出，所以拿不到 `CAMPPlus` 的模块定义。
从 checkpoint 反推前向（937 个张量、head 是 Conv2d 主干 + CAMDenseTDNN 块）能够做，
但**无法与参考实现对拍**，得到的"可分性/延迟"数字没有可信度，我不做这种交付。

两条可行路径：

1. **有外网的机器上导出 ONNX**（10 分钟）：
   ```python
   from funasr import AutoModel            # 或 3D-Speaker 的 CAMPPlus
   m = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common")
   torch.onnx.export(m.model, dummy_input, "campplus.onnx", opset_version=13)
   ```
   然后把 `campplus.onnx` 推到设备，用已有的 onnxruntime（ONNX 对照那时已经装过）跑，
   走"按行程注册 + 每段余弦"的门控，10–20 ms/段。
2. **换一个自带 ONNX 的说话人模型**（sherpa-onnx 的 3D-Speaker 导出包，20–30 MB，
   GitHub 上；本机访问不了）。

在这两条打通之前，声纹门控维持"设计完成、未验证"的状态；现有 TTS 统计声纹（§7）
已经证明不能上线，不要把它当作替代。

## 5. 复现

```sh
# 下载 SeACo 与分析模块（已验证）
cd ~/Library/Caches/qwen3-asr-htp/seaco
python3 - <<'PY'
import torch, collections
sd = torch.load('model.pt', map_location='cpu')
print(len(sd), 'tensors', sum(v.numel() for v in sd.values())/1e6, 'M params')
PY

# CAM++ 下载（已验证）与阻塞点（需外网 pip）
ls -la ~/Library/Caches/qwen3-asr-htp/speaker-models/campplus_cn_common.bin
```

## 30. M20：CAM++ 说话人门控（HTP+CPU 混合）

**路线**：ASR 留 HTP，声纹判定放 CPU，二者并行，互不占对方的尾延迟预算。

**导出**（`tools/export_campplus_onnx.py`）：用 funasr 1.4.9 的 `CAMPPlus` 加载
ModelScope 的 `campplus_cn_common.bin`（0 missing / 0 unexpected），导出 ONNX（27.0 MB，
输入 `feats (B,T,80)`，动态 T，opset 13）。注：本机 pip 走系统代理会重置连接，
最终是 curl 从阿里云镜像取 wheel、`--no-index` 本地安装才打通；librosa 需锁 0.10.x
（1.x 要求 Python ≥3.12）。

**已知偏差**：ONNX 与 PyTorch 的余弦相似度只有 **0.950/0.961**（不是 1e-5 级），说明导出图与
eager 前向存在数值路径差异。但门控判据是"相对距离"，实测在同一批数据上仍然完全可分（见下表），
所以先用它验证机制；要上线仍应把 parity 修到 0.999 以上（下一步）。

**门控实测**（`tools/spk_gate_eval.py`，Kaldi 风格 80 维 log-mel，司机=Tingting 16 条，
乘客=Flo 16 条，留一注册）：

| 指标 | 值 |
|---|---|
| 本人余弦 | 均值 0.914，最小 **0.838** |
| 冒充者余弦 | 均值 0.334，最大 **0.375** |
| 阈值 0.40–0.80 | FRR **0%**、FAR **0%** |
| EER 附近阈值 | 0.375（FRR 0% / FAR 0%） |

对比上一轮用 160 维 log-mel 统计做声纹（本人/冒充差距 0.011、噪声下余量 0.002）：
**预训练 embedding 的余量是 0.46，差了两个数量级**。这就是"要不要用预训练模型"的答案。

**成本**：CPU 上 2.5 s 音频 **118 ms**（21× 实时）。放在 HTP 之外并行跑，不进
`endpoint_to_final`；若要更快，可只用前 1 s（约 50 ms）或每段降采样。

**设备侧缺口**：板上只有 `libonnxruntime.so`，没有 `onnxruntime_c_api.h` 头文件，
要在 C++ 服务里跑门控需要补头文件（或用 sherpa-onnx 的封装）。这一步未做。

### 30.1 导出坑与修正（2026-11-18）

最初用 `dynamic_axes` 导出，ONNX 与 eager 的余弦只有 0.950。逐层对拍发现**每个子模块单独导出都是
1.000000**，问题出在整图导出对输入长度的敏感性。实测同一份权重、静态导出、三个不同输入的
最差余弦：

| 输入帧数 T | 最差余弦（3 个随机输入） |
|---:|---:|
| 100 | 0.571 |
| 200 | **1.000000** |
| 300 | 0.940 |
| 400 | **1.000000** |

也就是说：这个模型的 ONNX 图**只在部分长度上数值正确**，动态轴导出会静默返回错误 embedding
（这正是它"看起来还能 0/0，但余量虚高"的原因）。

修正方案：`export_campplus_onnx.py` 只导出 **T=200（2 s）固定长度**的图，导出后对 5 个随机输入
做 parity 校验（worst cosine 1.000000，失败即退出）；调用方把长音频切成 2 s 块、尾部用块均值
补齐、逐块取 embedding 后平均。

**修正后的门控实测**（同一批数据）：

| 指标 | 修正前（动态轴） | 修正后（定长 + 分块） |
|---|---|---|
| 本人余弦 | 0.914（最小 0.838） | **0.875（最小 0.793）** |
| 冒充者余弦 | 0.334（最大 0.375） | **0.416（最大 0.541）** |
| 安全阈值带 | 0.40–0.80 | **0.60–0.70**（FRR 0% / FAR 0%） |
| 余量 | 0.46（虚高） | **0.25（可信）** |
| 成本（2.5 s 片段） | 118 ms | 148 ms（两块） |

结论不变：**余量 0.25 比手写统计声纹的 0.011 大一个数量级以上**，门控机制成立；数值上现在可信。


## 32. M22：设备端说话人门控打通（不需要 onnxruntime 头文件）

**卡点的解决办法**：设备上有 `libsherpa-onnx-c-api.so` 与 `libonnxruntime.so`，而 sherpa 自带
`SherpaOnnxSpeakerEmbeddingExtractor`，它内部完成 fbank 与 onnxruntime 调用。只要给 ONNX 图补上
sherpa 要求的 metadata，就能在设备上直接跑我们的 CAM++，**完全不需要 `onnxruntime_c_api.h`**。

metadata 的两个坑（`tools/add_campplus_metadata.py`）：

1. sherpa 把 **`framework`** 当作 model type 校验（写成 "torch" 会报 `Unsupported model_type: torch`），
   必须写成受支持的家族（`3d-speaker` / `wespeaker` / `nemo`）；
2. 必须提供 **`output_dim`**（否则 `Create()` 直接 abort：`'output_dim' does not exist`）。

可用组合：`framework=3d-speaker, model_type=3d-speaker, sample_rate=16000, feature_dim=80,
normalize_samples=1, output_dim=192`。

**设备实测**（SM6650，sherpa 提取器 + onnxruntime，CPU）：

| 项 | 值 |
|---|---|
| 单段耗时（2.3 s 音频） | **107–123 ms** |
| 维度 | 192 |
| 本人余弦 | 均值 0.890，最小 **0.747** |
| 冒充者余弦 | 均值 0.373，最大 **0.445** |
| 安全阈值带 | **0.45–0.74**（FRR 0% / FAR 0%） |
| 分离余量 | **0.302** |

与宿主 Python 路径（余量 0.25）相比，设备端用 sherpa 自带的 fbank（含归一化）余量略大，两者都
远好于手写统计声纹的 0.011。**门控的"能跑"与"能分"现在都在真机上成立了**，剩下的只是接进服务。

**成本定位**：107–123 ms 在 CPU 上，与 HTP 上的 ASR（约 300 ms）并行时**不占尾延迟**；
下一批把它作为 `--speaker-gate`（启动时用 `--enroll-dir` 注册司机声纹，commit 时与 ASR 并发计算，
响应里返回 `speaker_score`）。
