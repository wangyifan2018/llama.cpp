# E37 全主干 Q8 的实际 greedy 投机与采样估算

2026-11-22。在 E35 全主干 Q8_0 mmproj 上，用同一台 SM6650、4 个固定 CPU worker 和 CPU Flash Attention 实测。回放帧使用 E35 的夹具，保留捕获的 Talker hidden 与 code0。首帧单独用作预热，后续 40 帧各跑两次，两个区块按相反顺序运行，合计每档 80 次记录、40 个独立输入。

窗口 2 原型实际生成 code。先用上一位置 hidden 和未来 head 产生 greedy 草稿，在同一张当轮图内验证两位置。草稿与目标 argmax 相同就提交草稿和 bonus。拒绝时只保留真实 code 和对应 KV 前缀。计时包括建图、分配、输入上传、计算、取 logits、接受判断与 KV 拷贝。每轮的 KV 在 host 回传并裁切，此原型没有生产环境图缓存和设备内 KV 管理。整帧 AR 也用同一个 harness 生成 code 和取 logits。没有计入 Talker、vocoder 或随机采样。

| 优化 CPU 路径 | 实测中位数 | 相对整帧 AR | 与整帧 AR 相同的完整帧 |
|---|---:|---:|---:|
| 整帧 greedy AR 图 | 84.56 ms | 1.000x | 80/80 |
| 逐轮 greedy AR，禁用投机 | 89.30 ms | 0.947x | 80/80 |
| 窗口 2，从 code8 开始 | 82.17 ms | **1.029x** | 56/80 |
| 窗口 2，从 code2 开始 | 85.68 ms | 0.987x | 48/80 |

从 code8 开始的两块中位数为 81.93 ms 和 82.40 ms，草稿接受 138/446。它比逐轮 AR 快约 8.7%，比现有整帧图只快约 2.9%，并出现可复现的 code 差异。逐轮 AR 与整帧 AR 在本组完全一致。

## 数值对照

固定首帧和每个已提交前缀时，优化 CPU 后端的两位置回放与单位置回放存在 logit 差异，平均绝对差 0.04827，最大 0.35254。首次验证的 15 个 argmax 仍一致。该数值差异在进入动态分轮与 KV 回退之前就已观察到。

打开 CPU reference 开关后，单、双、三位置固定帧 logits 逐元素相同。在 40 个帧输入上，分轮结果与各自 reference 整帧基线完全一致。从 code8 开始耗时 87.08 ms，reference 整帧 AR 为 84.98 ms。从 code2 开始为 85.14 ms。Reference 整帧与优化整帧在这 40 帧中只有 31 帧完全一致，所以它不能直接作为 E36 已部署分布的等价替代。该开关会影响多条优化路径，本轮没有隔离具体算子。

## 随机采样的估算

另在 Mac 上使用 Q8 模型采集 96 帧真实 AR 采样轨迹，top-k 50、top-p 0.8，计算实际草稿分布与目标分布的重合。code2、8、11、14 的接受率约 1.34%、42.77%、48.19%、45.65%。安卓 code8 的单步与两位置验证耗时为 5.70 和 6.52 ms。相比原精度对照，Q8 的两位置成本更靠近单步，但上述数值一致性仍需解决。

用实际 Q8 轨迹和安卓单轮计时模拟，从 code5 开始投机时，仅计计算，code2–15 约 **1.184x**。每个投机轮加 1 ms 额外开销后约 **1.070x**。这两个数来自估算，本轮没有把随机拒绝采样接入真实 TTS，也没有测合成首包、RTF 或音质。

## 复跑

先按 [E28](README.md) 构建 Mac 与 Android CPU 版本，按 [E35](quant-predictor.md) 预备全主干 Q8 文件、安卓库、首帧、40 帧夹具。

```sh
exp="$PWD/docs/backend/snapdragon/qwen3-tts-spec-experiment"
src=/tmp/llama-tts-spec-research
run=/tmp/qwen3tts-q8-spec-run
adb="$HOME/Library/Android/sdk/platform-tools/adb"
device=/data/local/tmp/q3tts-spec-quant
ndk_host="$HOME/Library/Android/sdk/ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64"
q8=/tmp/qwen3tts-quant-run/all-q8.gguf
original=/tmp/qwen3tts-spec-models
mkdir -p "$run/models"
bash "$exp/build-replay.sh" "$src" "$src/build-host"
CXX="$ndk_host/bin/aarch64-linux-android28-clang++" \
    bash "$exp/build-replay.sh" "$src" "$src/build-phone"
"$adb" push "$src/build-phone/bin/qwen3tts-spec-replay" "$device/q8-rounds"
"$adb" push "$exp/q8-spec-device.sh" "$device/"
"$adb" shell "cd $device && chmod 755 q8-rounds && sh q8-spec-device.sh"
for block in 1 2; do
    for mode in whole 16 8 2; do
        stem="q8-rounds-$mode-$block"
        "$adb" pull "$device/$stem.csv" "$device/$stem.log" "$device/$stem.codes" "$run/"
    done
done
"$adb" pull "$device/q8-window.csv" "$device/q8-window.scores" "$device/q8-window.log" "$run/"

"$adb" shell "cd $device && LD_LIBRARY_PATH=/data/local/tmp/q3tts-spec-20261122 \
    TTS_SPEC_PIN_MASK=f0 TTS_SPEC_FLASH=1 TTS_SPEC_CPU_REF=1 TTS_SPEC_WINDOW3=1 \
    taskset f0 ./q8-rounds all-q8.gguf frame0.bin q8-ref-window.scores 4 8 0 \
    >q8-ref-window.csv 2>q8-ref-window.log"
"$adb" pull "$device/q8-ref-window.csv" "$device/q8-ref-window.scores" "$device/q8-ref-window.log" "$run/"
for mode in whole 8 2; do
    stem="q8-ref-$mode"
    "$adb" shell "cd $device && LD_LIBRARY_PATH=/data/local/tmp/q3tts-spec-20261122 \
        TTS_SPEC_PIN_MASK=f0 TTS_SPEC_FLASH=1 TTS_SPEC_CPU_REF=1 TTS_SPEC_GREEDY_ROUNDS=$mode \
        taskset f0 ./q8-rounds all-q8.gguf phone-eval.bin $stem.codes 4 1 0 >$stem.csv 2>$stem.log"
    "$adb" pull "$device/$stem.csv" "$device/$stem.log" "$device/$stem.codes" "$run/"
done

ln -s "$original/qwen3-tts-0.6b-q8.gguf" "$run/models/qwen3-tts-0.6b-q8.gguf"
ln -s "$original/ref.wav" "$run/models/ref.wav"
ln -s "$q8" "$run/models/mmproj-qwen3-tts-0.6b-f16.gguf"
bash "$exp/capture.sh" "$src/build-host/bin/llama-tts" "$run/models" "$run/q8-capture"
TTS_SPEC_FLASH=1 TTS_SPEC_TRAIN_EXPORT="$run/heads.bin" "$src/build-host/bin/qwen3tts-spec-replay" \
    "$q8" "$run/q8-capture/frames.bin" "$run/q8-sampled.scores" 4 1 -1
python3 "$exp/q8-spec-summary.py" "$run"
```

[完整汇总](results/q8-spec-summary.txt) · [逐深度接受率与时延](results/q8-sampled-depths.csv) · [输入和输出散列](results/q8-spec.sha256)。原始逐帧计时、code、logits 和捕获在 `~/Desktop/qwen3-tts-spec-raw-20261122/q8-spec/`。
