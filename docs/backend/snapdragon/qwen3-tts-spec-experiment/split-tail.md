# E33 后段拆组与保留前九码 AR

2026-11-22。复用 E29 的 320 帧训练、160 帧留出和 96 帧回归输入、0.6B Base 以及单一参考音色。本轮重新用完整 greedy AR 生成对齐的老师 logits。第一组 code5–9 保持 E32 rank 32 权重。第二组把第一组采出的五个 code 作为一批 causal 输入，再用最后一位置的原 head 预测 code10，并行预测 code11–15。用 E32 相同的 rank 32 岭回归，分别对真实 AR 前缀和第一组自身前缀拟合第二组。

## 结果

| 留出 160 帧 | code10–15 重合度 | code11–15 重合度 |
|---|---:|---:|
| E32 单尾段 | 12.32% | 11.09% |
| 拆组，自身前缀，冻结第二组 | 10.44% | 9.36% |
| 拆组，自身前缀，用自身前缀训练第二组 | 12.12% | 11.37% |
| 拆组，真实 AR 前缀，冻结第二组 | 42.57% | 31.09% |
| AR 到 code9，使用真实前缀训练的第二组 | 44.38% | 33.25% |

E32 与拆组版的前九个 logits 在同一输入上逐元素一致。第一组对完整 greedy AR 的逐码 top-1 命中率依次为 100%、31.25%、26.25%、13.13%、10%，160 帧中没有一帧五个码全对。code10 的分布重合在自身前缀时只有 15.84%，在真实 AR 前缀下与老师逐元素一致。这解释了为何拆组增加一次计算后仍无整体收益。

| 安卓 SM6650 固定首帧，四线程绑定 | 30 次计算中位数 | 对同轮 AR |
|---|---:|---:|
| greedy AR | 184.85 ms | 1.00x |
| AR4 + 一个尾组 | 65.74 ms | 2.81x |
| AR4 + 两个尾组，自身前缀，已拟合 | 92.33 ms | 2.00x |
| AR9 + 一个尾组，已拟合 | 125.82 ms | 1.47x |

安卓同后端验证 AR9 的 code1–10 logits 与完整 greedy AR 一致。Mac 上 160 帧的 AR9 图与离线预期的后段 logits 最大绝对差约 0.00000334。手机计时单帧，每档预热 8 次，再计时 30 次，各档顺序执行。包含预测图与适配器计算，排除建图分配、输入上传、Talker、vocoder 和最终采样，未测完整合成或音质。重合度使用 top-k 50、top-p 0.8。

保持前九码 AR 能恢复一部分后段预测一致性，但加速从 E32 的 2.81x 降到 1.47x。若希望在此基础上继续提高质量，保留逐码依赖的小型 AR 学生更值得和 AR9 并行尾组做对照。用小样本先量教师强制和学生自由生成的差距，再决定是否投入更多训练。

## 复跑

先按 [E28](README.md) 构建 CPU 版本并部署安卓库和首帧，按 [E32](tail-student.md) 获取第一个适配器。继续使用 E29 的捕获和 E31 的留出帧。

```sh
exp="$PWD/docs/backend/snapdragon/qwen3-tts-spec-experiment"
src=/tmp/llama-tts-spec-research
run=/tmp/qwen3tts-split-tail-run
e32=/tmp/qwen3tts-tail-run
data=/tmp/qwen3tts-offset-run/offset-frames.bin
eval_frames=/tmp/qwen3tts-grouped-run/eval.bin
model=/tmp/qwen3tts-spec-models/mmproj-qwen3-tts-0.6b-f16.gguf
mkdir -p "$run"
bash "$exp/build-replay.sh" "$src" "$src/build-host"
replay="$src/build-host/bin/qwen3tts-spec-replay"

TTS_SPEC_FLASH=1 TTS_SPEC_GROUPED=ar-greedy "$replay" "$model" "$data" "$run/greedy.scores" 4 1 -1
TTS_SPEC_FLASH=1 TTS_SPEC_TAIL="$e32/results/tail.bin" TTS_SPEC_TAIL_SPLIT=raw \
    "$replay" "$model" "$data" "$run/raw.scores" 4 1 -1
python3 "$exp/split-tail.py" "$run" "$e32" --oracle-input "$data"
TTS_SPEC_FLASH=1 TTS_SPEC_TAIL="$e32/results/tail.bin" TTS_SPEC_TAIL_SPLIT=oracle \
    "$replay" "$model" "$run/oracle.bin" "$run/oracle.scores" 4 1 -1
OPENBLAS_NUM_THREADS=4 python3 "$exp/split-tail.py" "$run" "$e32" >"$run/summary.txt"

TTS_SPEC_FLASH=1 TTS_SPEC_TAIL="$e32/results/tail.bin" TTS_SPEC_TAIL_SPLIT="$run/second.bin" \
    "$replay" "$model" "$eval_frames" "$run/eval-adapted.scores" 4 1 -1
TTS_SPEC_FLASH=1 TTS_SPEC_TAIL="$e32/results/tail.bin" TTS_SPEC_TAIL_SPLIT="$run/second-oracle.bin" \
    TTS_SPEC_TAIL_AR9=1 "$replay" "$model" "$eval_frames" "$run/eval-ar9.scores" 4 1 -1
OPENBLAS_NUM_THREADS=4 python3 "$exp/split-tail.py" "$run" "$e32" --check-runtime --check-ar9
```

```sh
sdk="$HOME/Library/Android/sdk"
ndk_host="$sdk/ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64"
adb="$sdk/platform-tools/adb"
device=/data/local/tmp/q3tts-spec-20261122
device_mm=/data/local/tmp/q3tts/mmproj-qwen3-tts-0.6b-f16.gguf
CXX="$ndk_host/bin/aarch64-linux-android28-clang++" \
    bash "$exp/build-replay.sh" "$src" "$src/build-phone"
"$adb" push "$src/build-phone/bin/qwen3tts-spec-replay" "$device/qwen3tts-spec-split"
"$adb" push "$e32/results/tail.bin" "$device/e33-first.bin"
"$adb" push "$run/second.bin" "$device/e33-second.bin"
"$adb" push "$run/second-oracle.bin" "$device/e33-oracle.bin"
"$adb" shell "cd $device && chmod 755 qwen3tts-spec-split"
for mode in ar base split-raw split-fit ar9; do
    case "$mode" in
        ar) vars='TTS_SPEC_GROUPED=ar-greedy' ;;
        base) vars='TTS_SPEC_TAIL=e33-first.bin' ;;
        split-raw) vars='TTS_SPEC_TAIL=e33-first.bin TTS_SPEC_TAIL_SPLIT=raw' ;;
        split-fit) vars='TTS_SPEC_TAIL=e33-first.bin TTS_SPEC_TAIL_SPLIT=e33-second.bin' ;;
        ar9) vars='TTS_SPEC_TAIL=e33-first.bin TTS_SPEC_TAIL_SPLIT=e33-oracle.bin TTS_SPEC_TAIL_AR9=1' ;;
    esac
    "$adb" shell "cd $device && LD_LIBRARY_PATH=. TTS_SPEC_PIN_MASK=f0 TTS_SPEC_FLASH=1 $vars \
        taskset f0 ./qwen3tts-spec-split $device_mm frame0.bin e33-$mode.scores 4 30 0 \
        >e33-$mode.csv 2>e33-$mode.log"
    "$adb" pull "$device/e33-$mode.csv" "$device/e33-$mode.log" "$device/e33-$mode.scores" "$run/"
done
python3 "$exp/split-check.py" "$run"
```

[逐深度结果](results/split-tail-depths.csv) · [安卓计时汇总](results/split-tail-phone-check.txt) · [运行时与复跑核对](results/split-tail-runtime.txt) · [散列](results/split-tail.sha256)。原始 logits、安卓计时、日志与训练产物保存在本机 `~/Desktop/qwen3-tts-spec-raw-20261122/split-tail/`。
