# E30 双偏置与窗口 3

2026-11-22。复用 [E29](offset-distillation.md) 的 320 个训练帧、160 个留出帧、96 个回归帧。在 Mac M2 CPU 上拟合两个 1024 维共享偏置。第一个偏置和 E29 的 KL 产物逐字节相同。第二个偏置从同一 hidden 跨两级 RVQ 预测，冻结同一份 0.6B Base 老师和输出 head。两次拟合与评估合计 12.64 秒。

## 结果

| 留出集，起始深度 8–13 | 第一个草稿接受率 | 两个连续接受率 | 窗口 3 每轮预计提交 |
|---|---:|---:|---:|
| 两个原 head | 52.69% | 20.55% | 1.732 code |
| 只训练第一偏置 | 54.48% | 21.04% | 1.755 code |
| 两个偏置都训练 | 54.48% | 21.72% | 1.762 code |

| SM6650 CPU4–7，计算中位数 | 单步 | 两位置 | 三位置 |
|---|---:|---:|---:|
| code2 | 11.44 ms | 15.05 ms | 20.00 ms |
| code8 | 11.64 ms | 15.13 ms | 20.07 ms |
| code11 | 11.61 ms | 15.08 ms | 20.14 ms |
| code13 | 11.58 ms | 15.10 ms | 20.20 ms |

以 code8 为例，在当前成本下，三位置轮要胜过两位置轮，两个草稿的联合接受率要从实测 **19.08%** 升到约 **49.50%**。接受率保持现状时，三位置计算要从 **20.07 ms** 降到约 **17.04 ms** 才能追平。换设备可以用这两个门槛重新判断。

让模拟按深度选择窗口，限制在 2 和允许到 3 得到了同一个决策，保留前几步 AR，其余用窗口 2，最后一步回到 AR。两偏置模型在留出集上估计 **1.111×**，每个投机轮加 1 ms 额外开销后为 **1.058×**。回归集对应 **1.103×** 和 **1.054×**。这些是 code2–15 的模拟比值，整帧图、采样、KV 提交、首帧和端到端 RTF 仍未实测。

## 测量口径

三位置验证包含两个草稿 head 和三个目标 head。硬件 microbench 复用 E28 的固定首帧和 AR 前缀 KV，CPU Flash Attention，四线程分别固定到 CPU4–7。每个深度和可用窗口测 6 块，每块热身 3 次、计时 12 次，以轮转顺序执行。三种窗口在该帧的目标 logits 逐元素相同。计时排除采样、偏置加法、输入准备和 KV 提交。

联合接受率用 AR 轨迹估计。给定首个真实输出 code x，条件接受概率是 min(1,q1(x)/p1(x))。乘以后续真实前缀下 p2 和 q2 的分布重合度，再对轨迹求平均。这样保留了第二步目标分布依赖第一个码的条件关系。双偏置草稿在同一 hidden 上分别计算，第二个草稿分布不随第一个候选变化。调度策略用训练集概率选窗口，评估时沿留出或回归轨迹用各位置的条件接受概率计算期望开销。目标分布为 top-k 50、top-p 0.8、temperature 1。

[逐深度接受率](results/window3-depths.csv) · [安卓原始计时](results/window3-phone-timing.csv) · [调度结果](results/window3-summary.txt) · [复跑核对](results/window3-replay-check.txt) · [文件散列](results/window3.sha256)

## 复现

先按 [E28](README.md) 构建 Mac 和 Android 的 CPU 版本，按 [E29](offset-distillation.md) 获取 `offset-frames.bin`。以下命令使用 E28 已部署的模型、共享库和首帧。

```sh
exp="$PWD/docs/backend/snapdragon/qwen3-tts-spec-experiment"
src=/tmp/llama-tts-spec-research
models=/tmp/qwen3tts-spec-models
data=/tmp/qwen3tts-offset-run
run=/tmp/qwen3tts-window3-run
regression="$HOME/Desktop/qwen3-tts-spec-raw-20261122/frames.bin"
sdk="$HOME/Library/Android/sdk"
ndk_host="$sdk/ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64"
adb="$sdk/platform-tools/adb"
device=/data/local/tmp/q3tts-spec-20261122
device_mm=/data/local/tmp/q3tts/mmproj-qwen3-tts-0.6b-f16.gguf

mkdir -p "$run"
bash "$exp/build-replay.sh" "$src" "$src/build-host"
CXX="$ndk_host/bin/aarch64-linux-android28-clang++" \
    bash "$exp/build-replay.sh" "$src" "$src/build-phone"

TTS_SPEC_FLASH=1 TTS_SPEC_WINDOW3=1 TTS_SPEC_TRAIN_EXPORT="$run/heads.bin" \
    "$src/build-host/bin/qwen3tts-spec-replay" "$models/mmproj-qwen3-tts-0.6b-f16.gguf" \
    "$data/offset-frames.bin" "$run/window3.scores" 4 1 -1 >"$run/export.csv" 2>"$run/export.log"

"$adb" push "$src/build-phone/bin/qwen3tts-spec-replay" "$device/qwen3tts-spec-window3"
"$adb" shell "cd $device && chmod 755 qwen3tts-spec-window3 && \
    LD_LIBRARY_PATH=. TTS_SPEC_PIN_MASK=f0 TTS_SPEC_FLASH=1 TTS_SPEC_WINDOW3=1 \
    taskset f0 ./qwen3tts-spec-window3 $device_mm frame0.bin w3-phone.scores 4 12 0 \
    >w3-phone.csv 2>w3-phone.log"
"$adb" pull "$device/w3-phone.csv" "$device/w3-phone.log" "$device/w3-phone.scores" "$run/"
OPENBLAS_NUM_THREADS=4 python3 "$exp/window3.py" "$data" "$run/window3.scores" \
    "$run/heads.bin" "$run/w3-phone.csv" "$run/results" --regression "$regression" >"$run/summary.txt"
```

原始导出、安卓日志和训练偏置保存在本机 `~/Desktop/qwen3-tts-spec-raw-20261122/window3/`。
