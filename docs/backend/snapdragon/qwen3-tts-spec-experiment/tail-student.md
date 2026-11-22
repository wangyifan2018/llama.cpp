# E32 前四码 AR 加低秩后段

2026-11-22。沿用 E29 的 320 帧训练、160 帧留出和 96 帧回归数据，同一个 Qwen3-TTS 0.6B Base checkpoint 和参考音色。E31 第一阶段的前缀误差由现有 AR 路径消除。实验检查恢复前缀后，低秩后段能否达到更接近原模型的预测。

## 实测

| SM6650 单帧图计算，CPU4–7，每个 worker 单独固定 | 中位数 | 最小–最大 |
|---|---:|---:|
| greedy AR | 176.56 ms | 174.70–189.78 ms |
| 前四码 greedy AR，后段 rank 32 | 62.83 ms | 62.54–64.88 ms |

图计算比值 **2.81×**。两档使用同一固定首帧，分别预热 8 次、计时 30 次，依次执行。计时包括 rank 32 适配器的投影、十个修正矩阵乘和加法，以及前四码 argmax 与反馈。计时排除建图、分配、输入上传、后段采样、Talker 反馈 embedding、vocoder 和 PCM 输出。安卓和 Mac 各自的 code1–5 logits 与对应 CPU backend 的 greedy AR 逐元素一致。Mac 的后段运行时与 NumPy 对照，最大绝对 logit 差为 0.00000525，全部 argmax 一致。

| 留出 160 帧，code6–15 | 同前缀冻结头 | 共享 KL 偏置 | rank 32 |
|---|---:|---:|---:|
| 原采样 AR 前缀，分布重合 | 17.93% | 19.62% | **21.39%** |
| 原采样 AR 前缀，top-1 一致率 | 9.81% | 10.94% | **12.13%** |
| greedy AR 前缀，分布重合 | 16.68% | 18.17% | **19.89%** |
| greedy AR 前缀，top-1 一致率 | 7.69% | 9.38% | **10.81%** |

code5 是精确前缀下的原头，排除在这张后段表之外。greedy 留出集上，rank 8、32、64 的分布重合率分别为 **18.44%、19.89%、20.23%**。rank 32 是预先选的设备计时配置，其他秩只做离线对照。这个小样本实验支持前缀控制与图成本的判断。后段对原模型分布的匹配仍弱，完整合成速度、自由生成稳定性和音质未测。

## 方法

前四码使用原 predictor 自回归生成。用第四码更新前缀后，用原 head 计算 code5 logits。用训练集上的 PCA 取其前 32 个白化特征。对 code6–15 各训练一个岭回归头，以原 AR teacher logits 与 E31 共享偏置学生 logits 的差为目标。目标差逐头去掉词表均值，岭惩罚为训练帧数乘 0.5。在线把预测修正加到冻结 head 上，PCA 的均值已折进修正偏置。冻结老师和学生主干均来自同一个 checkpoint。Python/NumPy 拟合与评估用时 5.14 秒。

分布重合使用 top-k 50、top-p 0.8。greedy 对照中双方前四码生成方式一致，之后的原 AR 仍逐码条件化，学生后段全部依赖同一个锚点。

## 复现

先按 [E28](README.md) 构建并在安卓部署相关库与首帧，按 [E31](grouped.md) 生成老师、greedy 前缀和偏置。沿用 E29 的捕获和散列。

```sh
exp="$PWD/docs/backend/snapdragon/qwen3-tts-spec-experiment"
src=/tmp/llama-tts-spec-research
e29=/tmp/qwen3tts-offset-run
e31=/tmp/qwen3tts-grouped-run
run=/tmp/qwen3tts-tail-run
model=/tmp/qwen3tts-spec-models/mmproj-qwen3-tts-0.6b-f16.gguf
sdk="$HOME/Library/Android/sdk"
ndk_host="$sdk/ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64"
adb="$sdk/platform-tools/adb"
device=/data/local/tmp/q3tts-spec-20261122
device_mm=/data/local/tmp/q3tts/mmproj-qwen3-tts-0.6b-f16.gguf

mkdir -p "$run"
OPENBLAS_NUM_THREADS=4 python3 "$exp/tail-student.py" "$e29" "$e31" "$run/results" >"$run/summary.txt"
bash "$exp/build-replay.sh" "$src" "$src/build-host"
CXX="$ndk_host/bin/aarch64-linux-android28-clang++" \
    bash "$exp/build-replay.sh" "$src" "$src/build-phone"
TTS_SPEC_FLASH=1 TTS_SPEC_TAIL="$run/results/tail.bin" "$src/build-host/bin/qwen3tts-spec-replay" \
    "$model" "$e31/eval.bin" "$run/eval-tail.scores" 4 1 -1 >"$run/host.csv" 2>"$run/host.log"

"$adb" push "$src/build-phone/bin/qwen3tts-spec-replay" "$device/qwen3tts-spec-tail"
"$adb" push "$run/results/tail.bin" "$device/tail.bin"
"$adb" shell "cd $device && chmod 755 qwen3tts-spec-tail && \
  export LD_LIBRARY_PATH=. TTS_SPEC_PIN_MASK=f0 TTS_SPEC_FLASH=1 && \
  TTS_SPEC_GROUPED=ar-greedy taskset f0 ./qwen3tts-spec-tail $device_mm frame0.bin e32-ar.scores \
    4 30 0 >e32-ar.csv 2>e32-ar.log && \
  TTS_SPEC_TAIL=tail.bin taskset f0 ./qwen3tts-spec-tail $device_mm frame0.bin e32-tail.scores \
    4 30 0 >e32-tail.csv 2>e32-tail.log"
for mode in ar tail; do
    "$adb" pull "$device/e32-$mode.csv" "$device/e32-$mode.log" "$device/e32-$mode.scores" "$run/"
done
python3 "$exp/tail-check.py" "$run" "$e31"
```

改变 `tail-student.py --rank` 可复跑秩对照。原始 logits、计时、日志和模型文件保存在 `~/Desktop/qwen3-tts-spec-raw-20261122/tail-student/`。

[逐深度结果](results/tail-depths.csv) · [秩对照](results/tail-rank-summary.txt) · [运行时核对](results/tail-runtime-check.txt) · [安卓 AR 计时](results/tail-phone-ar.csv) · [安卓适配器计时](results/tail-phone-adapter.csv) · [复跑](results/tail-repeat.txt) · [散列](results/tail.sha256)
