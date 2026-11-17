# 座舱降噪方案调研：能拿到什么、值不值得上（2026-11-18）

范围：单通道降噪（SE）、语音分离（SS）、目标说话人提取（TSE）、多麦波束成形，
以及它们在本工程（Paraformer-zh/en，SM6650 HTP，endpoint->final 277 ms）里的位置。

## 0. 结论摘要

1. **单通道降噪不要进 `endpoint_to_final` 关键路径**：实测白噪声上降噪拿到 +3.7~5.6 dB，
   命令字段一点没涨、10 dB 时还掉 6 点；而稳态噪声本身对模型影响很小（§18.3/§20）。
   ZipEnhancer 是非因果的（需要看未来帧），给它 200–300 ms 前瞻等于直接把尾延迟打上去。
2. **真正打崩识别的是另一路说话人（babble）**：5 dB 时命令字段从 56.2% 掉到 31.2%。
   这一类单通道降噪无能为力，需要**说话人相关**的处理。
3. **ModelScope 上没有任何现成的目标说话人提取（TSE）模型**；最接近的是 MossFormer2 分离
   （8 kHz、213 MB），但它分的是"两个人",不是"我要的那个人"，而且 RTF 大于 1，
   放不进 30 ms 的尾延迟余量。
4. 能立刻用、且成本可接受的是 **CAM++ 说话人向量（27 MB）**：做"按行程注册 + 门控"，
   判一次约 µs 级（统计特征）/ 10–20 ms（ONNX 模型），用来挡乘客，而不是做分离。
5. **多麦波束成形**才是车载降噪的正解，但它依赖硬件通道数与阵列几何，不是模型问题。

## 1. ModelScope 实测可用性（2026-11-18 探测）

| 模型 | 大小 | ONNX | 类别 | 备注 |
|---|---:|---|---|---|
| `iic/speech_zipenhancer_ans_multiloss_16k_base` | 17.4 MB | **有** | 单通道降噪 | 16 kHz，最轻，非因果 |
| `iic/speech_frcrn_ans_cirm_16k` | 55.3 MB | 无 | 单通道降噪 | FunASR 经典，需 torch→onnx |
| `iic/speech_dfsmn_ans_psm_48k_causal` | 5.8 MB | 无 | 单通道降噪（因果） | 48 kHz，延迟最低 |
| `iic/speech_mossformer2_separation_temporal_8k` | 213.1 MB | 无 | 语音分离（2 人） | **8 kHz**，RTF > 1 |
| `iic/speech_mossformer_separation_temporal_8k` | 160.8 MB | 无 | 语音分离（2 人） | 同上 |
| `iic/speech_campplus_sv_zh-cn_16k-common` | 27.1 MB | 无 | 说话人向量 | 16 kHz，CAM++ |
| `iic/speech_eres2netv2_sv_zh-cn_16k-common` | 68.4 MB | 无 | 说话人向量 | 更强，更大 |
| `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` | 282.5 MB | 无 | 标点 | 可替代停顿启发式 |
| `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 1.7 MB | 无 | VAD | 若要把 VAD 搬进服务端 |
| `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 952.2 MB | 无 | ASR + 上下文偏置 | 自带热词机制 |
| 目标说话人提取（TSE） | — | — | — | **未找到**（`speech_3d_speaker`、mossformer2-16k 均 404） |

## 2. 四类方案与它们的真实代价

### 2.1 单通道降噪（ZipEnhancer / FRCRN / DFSMN）

- **收益**：只对稳态噪声有效（路噪、空调风）。我们的实测：这类噪声本来就不怎么影响识别
  （粉红噪声 5 dB 时中文 CER 仍是 0），所以收益接近 0。
- **代价**：非因果模型 200–300 ms 前瞻；因果模型（DFSMN）延迟低但降噪能力弱。
  在 277 ms 的总延迟里，300 ms 前瞻是**不可接受**的。
- **结论**：不上。若将来真实录音证明某种噪声会掉字，优先试 `DFSMN_ANS_48k_causal`
  （5.8 MB、因果），并且必须在命令字段上验收，而不是 SNR/PESQ。

### 2.2 语音分离（MossFormer2 8k）

- 把混合音分成两路，但**分出来的是"说话人 A / 说话人 B"，不是"司机"**；
- 8 kHz 会先毁掉 16 kHz 特征（我们的 fbank 是 16 kHz）；213 MB 权重；RTF > 1，
  2.7 s 音频要 2.7 s 以上算完——**尾延迟直接翻倍**。
- **结论**：不能上设备。可作为**离线数据增强**工具（给训练/评测造带人声干扰的样本）。

### 2.3 目标说话人提取（TSE）

- 理想形态：注册司机声纹 → 单模型输出"司机那一路"。VoiceFilter / SpEx+ 一类。
- **ModelScope 上没有现成模型**；用"MossFormer2 分离 + CAM++ 选路"能拼出来，但等于把
  2.2 的代价全背上，再叠加选路误差。
- **结论**：现阶段不做。若要做，只有两条路：自己训（需要成对数据），或等开源权重。

### 2.4 多麦波束成形

- 不需要模型，延迟 ~1 ms，对**方向性干扰**（旁边座位的人）最有效；
- 依赖硬件：麦克风数量、阵列几何、通道同步。座舱通常有 2–4 个麦，但我们目前只拿到单通道。
- **结论**：产品侧最该推动的方向；软件能做的只是把多通道接进来做 delay-and-sum / MVDR。

## 3. 与本工程指标的关系

| 方案 | 额外延迟 | 对 endpoint->final 的影响 | 对车控精度的影响 |
|---|---:|---|---|
| ZipEnhancer（非因果） | +200–300 ms | **致命** | 稳态噪声下 ≈0 |
| DFSMN 因果降噪 | ~1 ms/帧 | 可接受 | 稳态噪声下 ≈0 |
| MossFormer2 分离 | RTF > 1 | **致命** | babble 下可能有用 |
| CAM++ 声纹门控 | ~0（统计）/ 10–20 ms（ONNX） | 可接受 | 减少乘客误触发 |
| 多麦波束成形 | ~1 ms | 可接受 | babble 下最有效 |

## 4. 建议

1. **短期（不改变尾延迟）**：只做 CAM++ 声纹门控（按行程注册），挡乘客误触发。
   注意 §7 的结论：TTS 声纹不能直接上线，必须换成预训练 embedding + 真实录音验证。
2. **中期（需要硬件）**：推动多麦接入，做波束成形；这是"降噪"在车内的正确解法。
3. **离线价值**：MossFormer2 分离用来造"有人说话干扰"的评测集，比上线更划算。
4. **不做**：把非因果降噪塞进 `endpoint_to_final`；把 8 kHz 分离模型上设备；
   在没有 TSE 权重的情况下自己拼"分离+选路"。
5. 标点模型（CT-Transformer 282 MB）可以作为"停顿标点"的替代验证项，但当前停顿法已经
   把命令字段做到 78.1%，优先级低于上下文偏置。

## 5. 复现

```sh
# 探测 ModelScope 上模型是否存在、多大、有没有 ONNX
python3 - <<'PY'
import json, urllib.request
for mid in ["iic/speech_zipenhancer_ans_multiloss_16k_base", "iic/speech_campplus_sv_zh-cn_16k-common"]:
    d = json.load(urllib.request.urlopen("https://modelscope.cn/api/v1/models/%s/repo/files?Revision=master" % mid, timeout=30))
    files = d["Data"]["Files"]
    print(mid, sum(f["Size"] for f in files if f["Type"] != "tree") / 1048576.0, "MB")
PY
```

本机网络：`huggingface.co` 不可达，`modelscope.cn` 可达（2026-11-18 实测）。
