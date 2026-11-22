# E35 五层 AR predictor 的权重量化

2026-11-22。保留五层、逐码反馈、head、embedding、Talker 和 vocoder。原始 mmproj 文件名带 F16，但 predictor 的 5 个 FFN down 矩阵为 F32，其余 30 个主干矩阵为 F16。用 GGUF writer 构造四个对照，通过读回校验元数据、张量名称、形状、类型及数据。每档目标矩阵以外的所有数据逐元素相同。全主干 Q8 GGUF 独立生成两次，逐字节一致。

| 权重配置 | 安卓固定帧计算中位数 | 加速 | Mac 同前缀重合 | 安卓自由反馈重合 | 安卓 15 码全一致 |
|---|---:|---:|---:|---:|---:|
| 原始混合精度 | 184.91 ms | 1.00x | 100% | 100% | 40/40 |
| FFN down 改 F16 | 157.52 ms | 1.17x | 99.98% | 99.98% | 40/40 |
| 仅 FFN down 改 Q8_0 | 142.97 ms | 1.29x | 97.66% | 84.26% | 26/40 |
| 其余 30 个矩阵 Q8_0，FFN down 保留 F32 | 126.63 ms | 1.46x | 96.30% | 77.36% | 17/40 |
| 全部 35 个矩阵 Q8_0 | 84.46 ms | **2.19x** | 95.65% | 76.67% | 16/40 |

全主干 Q8 的 35 个矩阵从 180 MiB 降至 79.69 MiB，减少约 100.31 MiB。Mac 在老师前缀下的 top-1 一致率为 93.46%，安卓帧内自由反馈的逐码一致率为 73.17%。同前缀与自由反馈的差距与逐码误差传播一致，但两组分别使用不同平台和样本数，差值不能直接当成传播的精确幅度。全主干 Q8 在本轮手机自由反馈的速度与码一致性组合上更有吸引力，是否值得采用需要听评。原 [转换器](../../../../conversion/qwen3tts.py) 保留 FFN down 的 F32 是为避免 FFN 激活转 F16 时溢出。down-f16 对照在本批次的输出有限且行为接近原版，但未覆盖长文本或高幅度激活，不能据此移除原保护。

Mac teacher forcing 使用 160 个留出帧和额外 96 个回归帧，原版回放与 E33 原 greedy logits 逐元素一致。安卓帧内自由反馈使用四条留出短句各首 10 帧，共 40 帧，所有配置使用同一帧输入；Talker hidden 每帧固定为捕获值。重合度使用 top-k 50、top-p 0.8。两边都没有在跨帧 Talker 反馈回路里运行量化模型。

安卓 SM6650 CPU4–7 逐线程固定，CPU Flash Attention，固定 E28 首帧。每个配置两块，每块在一次初始计算和 8 次预热后计时 24 次，第一批四个配置按正序和反序执行，保留 FFN down 的对照随后单独补跑。全主干 Q8 两块的时延中位数分别为 84.33 和 84.75 ms。计时排除建图、分配、输入上传、Talker、vocoder 和音频写出，未测完整合成时延或音质。

## 复跑

先按 [E28](README.md) 构建 Mac 和 Android 的 CPU 版本并部署安卓运行库，按 [E34](small-ar.md) 取得老师前缀输入。使用 Python 3.12、NumPy、PyYAML 和仓库的 gguf-py。下面的 PYTHONPATH 附加路径是本次实验现有的 PyYAML 安装目录。

```sh
exp="$PWD/docs/backend/snapdragon/qwen3-tts-spec-experiment"
src=/tmp/llama-tts-spec-research
run=/tmp/qwen3tts-quant-run
source=/tmp/qwen3tts-spec-models/mmproj-qwen3-tts-0.6b-f16.gguf
eval=/tmp/qwen3tts-small-ar-run/eval.bin
frames="$HOME/Desktop/qwen3-tts-spec-raw-20261122/frames.bin"

mkdir -p "$run"
PYTHONPATH="$PWD/gguf-py:/tmp/pylibs" python3 "$exp/quant-predictor.py" "$source" "$run" >"$run/conversion.txt"
python3 "$exp/quant-summary.py" "$run" --prepare "$frames" "$eval"
for mode in original down-f16 down-q8 all-q8 keep-down-q8; do
    model="$run/$mode.gguf"
    if [ "$mode" = original ]; then model="$source"; fi
    TTS_SPEC_FLASH=1 TTS_SPEC_GROUPED=ar-forced "$src/build-host/bin/qwen3tts-spec-replay" \
        "$model" "$eval" "$run/host-$mode.scores" 4 1 -1 >"$run/host-$mode.csv" 2>"$run/host-$mode.log"
done
adb="$HOME/Library/Android/sdk/platform-tools/adb"
device=/data/local/tmp/q3tts-spec-quant
bash "$exp/quant-phone.sh" "$adb" "$src/build-phone/bin/qwen3tts-spec-replay" "$run" "$device"
QUANT_KEEP_DOWN_ONLY=1 bash "$exp/quant-phone.sh" "$adb" "$src/build-phone/bin/qwen3tts-spec-replay" "$run" "$device"
python3 "$exp/quant-summary.py" "$run"
```

[完整结果](results/quant-predictor-summary.txt) · [逐深度数据](results/quant-predictor-depths.csv) · [转换校验](results/quant-predictor-conversion.txt) · [重复转换](results/quant-predictor-repeat.txt) · [GGUF 与回放输入散列](results/quant-predictor-inputs.sha256)。原始 logits、时延和日志保存在本机 `~/Desktop/qwen3-tts-spec-raw-20261122/quant-predictor/`。生成的 GGUF 位于上述 run 目录，也已推送到手机实验目录。
