# E34 三层自回归 predictor 与轻量蒸馏

2026-11-22。保留 Qwen3-TTS 0.6B Base predictor 的原 code embedding、code head、逐码反馈与 RVQ 顺序。原 predictor 有五层，三层试验保留零基编号 0、2、4。其余权重冻结。使用 E33 的完整 greedy AR 轨迹作老师，320 帧拟合，160 帧留出，96 帧回归，单一参考音色。

蒸馏只校正 logits。用训练数据的 pruned logits 拟合共享 rank 24 的 PCA 投影，再为每个码本拟合一个带截距的岭回归，将原老师与三层学生的 logit 差作为目标。拟合与首轮评估用时 2.94 秒。老师前缀和自由反馈分开评估，分布重合采用 top-k 50、top-p 0.8。

## 结果

| 安卓 SM6650 四线程绑核，同一固定帧 | 30 次图计算中位数 | 相对速度 |
|---|---:|---:|
| 五层 greedy AR | 183.07 ms | 1.00x |
| 三层 greedy AR | 111.06 ms | 1.65x |
| 三层 greedy AR 加 rank 24 校正 | 113.45 ms | 1.61x |

| 留出 160 帧 | 分布重合度 | 逐码 top-1 |
|---|---:|---:|
| 三层，老师前缀 | 10.20% | 3.83% |
| 三层加校正，老师前缀 | 14.32% | 7.29% |
| 三层，自由反馈 | 5.45% | 2.29% |
| 三层加校正，自由反馈 | 8.30% | 3.88% |

加入校正后，留出集首码 top-1 从 21.88% 提到 36.88%，但 160 帧中无一帧的前四码全部匹配原 AR。相同校正的回归集首码一致率从 30.21% 降到 22.92%，小样本结果不稳定。原权重裁层对照保留 0、1、4 时老师前缀重合度为 2.98%，保留 0、1、3、4 时为 4.76%。本轮仅覆盖这些组合，数据说明保留层数和保留位置都会影响结果。

手机两组原始图与校正图都走 CPU Flash Attention。每档预热 8 次、计时 30 次，顺序执行。耗时包含逐码 greedy 反馈和校正计算，排除建图、分配、输入上传、Talker、vocoder 与音频输出。校正的 Mac 图与 NumPy 离线结果逐位置 argmax 一致，最大绝对 logit 差约 0.00000572。自由反馈的指标衡量对原轨迹的接近程度，未测音质和完整合成速度。

## 复跑

先按 [E28](README.md) 配置 Mac 和 Android 的构建与设备部署，使用 [E33](split-tail.md) 的 greedy 老师 logits。手机使用 E28 的首帧。

```sh
exp="$PWD/docs/backend/snapdragon/qwen3-tts-spec-experiment"
src=/tmp/llama-tts-spec-research
run=/tmp/qwen3tts-small-ar-run
model=/tmp/qwen3tts-spec-models/mmproj-qwen3-tts-0.6b-f16.gguf
input=/tmp/qwen3tts-offset-run/offset-frames.bin
teacher=/tmp/qwen3tts-split-tail-run/greedy.scores
mkdir -p "$run"
bash "$exp/build-replay.sh" "$src" "$src/build-host"
replay="$src/build-host/bin/qwen3tts-spec-replay"

python3 "$exp/small-ar.py" "$run" "$teacher" --capture "$input"
TTS_SPEC_FLASH=1 TTS_SPEC_GROUPED=ar-forced TTS_SPEC_LAYERS=024 \
    "$replay" "$model" "$run/teacher-prefix.bin" "$run/forced.scores" 4 1 -1
TTS_SPEC_FLASH=1 TTS_SPEC_GROUPED=ar-greedy TTS_SPEC_LAYERS=024 \
    "$replay" "$model" "$run/eval.bin" "$run/free.scores" 4 1 -1
OPENBLAS_NUM_THREADS=4 python3 "$exp/small-ar.py" "$run" "$teacher"

TTS_SPEC_FLASH=1 TTS_SPEC_GROUPED=ar-forced TTS_SPEC_LAYERS=024 TTS_SPEC_AR_CORRECTION="$run/correction.bin" \
    "$replay" "$model" "$run/eval.bin" "$run/forced-fit.scores" 4 1 -1
TTS_SPEC_FLASH=1 TTS_SPEC_GROUPED=ar-greedy TTS_SPEC_LAYERS=024 TTS_SPEC_AR_CORRECTION="$run/correction.bin" \
    "$replay" "$model" "$run/eval.bin" "$run/free-fit.scores" 4 1 -1
for layers in 014 0134; do
    TTS_SPEC_FLASH=1 TTS_SPEC_GROUPED=ar-forced TTS_SPEC_LAYERS="$layers" \
        "$replay" "$model" "$run/eval.bin" "$run/forced-$layers.scores" 4 1 -1
done
```

```sh
sdk="$HOME/Library/Android/sdk"
ndk_host="$sdk/ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64"
adb="$sdk/platform-tools/adb"
device=/data/local/tmp/q3tts-spec-20261122
device_mm=/data/local/tmp/q3tts/mmproj-qwen3-tts-0.6b-f16.gguf
CXX="$ndk_host/bin/aarch64-linux-android28-clang++" \
    bash "$exp/build-replay.sh" "$src" "$src/build-phone"
"$adb" push "$src/build-phone/bin/qwen3tts-spec-replay" "$device/qwen3tts-spec-small"
"$adb" push "$run/correction.bin" "$device/e34-correction.bin"
"$adb" shell "cd $device && chmod 755 qwen3tts-spec-small"
for mode in ar small fit; do
    case "$mode" in
        ar) vars='' ;;
        small) vars='TTS_SPEC_LAYERS=024' ;;
        fit) vars='TTS_SPEC_LAYERS=024 TTS_SPEC_AR_CORRECTION=e34-correction.bin' ;;
    esac
    "$adb" shell "cd $device && LD_LIBRARY_PATH=. TTS_SPEC_PIN_MASK=f0 TTS_SPEC_FLASH=1 \
        TTS_SPEC_GROUPED=ar-greedy $vars taskset f0 ./qwen3tts-spec-small $device_mm frame0.bin \
        e34-$mode.scores 4 30 0 >e34-$mode.csv 2>e34-$mode.log"
    "$adb" pull "$device/e34-$mode.csv" "$device/e34-$mode.log" "$device/e34-$mode.scores" "$run/"
done
OPENBLAS_NUM_THREADS=4 python3 "$exp/small-ar.py" "$run" "$teacher" --check-runtime --android --ablation
```

[结果](results/small-ar-summary.txt) · [逐深度指标](results/small-ar-depths.csv) · [重复拟合与数值校验](results/small-ar-check.txt) · [散列](results/small-ar.sha256)。原始 logits、计时和适配器位于 `~/Desktop/qwen3-tts-spec-raw-20261122/small-ar/`。
