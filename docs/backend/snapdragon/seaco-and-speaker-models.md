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
