# E31 两阶段 4+11 predictor 原型

2026-11-22。复用 E29 的同一个 0.6B Base checkpoint，以及 320 个训练帧、160 个留出帧和 96 个回归帧。主干、code embedding 和原 head 均冻结。这是用两次宽前向检查性能和前缀误差的原型。

## 结果

| SM6650，固定首帧，四线程逐核绑定 | 次数 | 中位数 | 最小–最大 |
|---|---:|---:|---:|
| greedy AR 整帧 predictor 图 | 30 | 180.69 ms | 178.19–183.21 ms |
| 4+11 两阶段，第二阶段读取前段自身生成的 code | 30 | 40.01 ms | 39.44–40.60 ms |

两阶段图计算时间约快 **4.52×**。两组都用 CPU Flash Attention，执行前各预热 8 次。第一阶段三个修正后的 head 加偏置后取 argmax。两阶段图包含这三次修正和四个 argmax，后段共享偏置在离线评估时应用。计时排除建图、分配、Talker、vocoder 和输出拷贝。每档来自同一固定首帧的一组计时，未交错调度。完整合成速度和音质未测。

| 留出 160 帧，code5–15 对原采样 AR 的分布重合 | 重合度 |
|---|---:|
| 原采样 AR 前缀，后段使用原冻结头 | 25.40% |
| 原采样 AR 前缀，后段使用 KL 偏置 | 26.93% |
| 自身 greedy 前缀，使用老师前缀训练的后段偏置 | 4.45% |
| 自身 greedy 前缀，后段改用自身前缀训练 | 4.98% |

因为原 AR 使用随机采样，换成 greedy 本来就可能走另一条合法轨迹。补做了完整 AR greedy 对照。第一阶段与完整 AR greedy 的逐码匹配率从 code1 到 code4 分别为 **100.00%、4.38%、0.00%、0.63%**，160 帧中无完整前缀匹配。对齐 greedy 参考后，后段在正确 greedy 前缀下分布重合为 **24.25%**，换成原型前缀后为 **4.36%**。在两种老师前缀各自的轨迹上，原型的 code5 logits 与原 AR 逐元素相同。控制结果把前四码的并行预测误差与采样轨迹差异分开。

这个结果支持继续研究缩短串行路径。当前原型的第一阶段共享 hidden 加偏置，对 code2–4 的预测太弱。下一种值得验证的结构是保留短的前段 AR，再训练显式读取前段 code 的后段学生。后段 oracle 前缀的重合度也只有约 25%，需要比共享偏置更强的学生，音质应靠自由生成和听评判断。

## 方法与复现

第一阶段一次 causal 前向处理 Talker hidden 与 code0 embedding，用最后位置 hidden 预测 code1–4。保留 code1 原头，给 code2–4 的草稿拟合一个 1024 维共享 KL 偏置。第二阶段复用前缀 KV，一次 causal 前向读取 code1–4，用最后位置 hidden 预测 code5–15。code5 原头保持冻结，针对 code6–15 各拟合一个共享偏置对照，分别使用原采样 AR 前缀和第一阶段生成的 greedy 前缀。每个偏置从零初始化，全批 Adam 120 步，学习率 0.01。分布重合使用 E29 的 top-k 50、top-p 0.8。与另一路径的理论接受率和音质分别属于不同问题，这里报告分布重合。

先按 [E28](README.md) 构建 CPU 版本并部署安卓依赖，按 [E29](offset-distillation.md) 生成数据。以下使用已采集的 E29 输入和 E28 手机首帧。

```sh
exp="$PWD/docs/backend/snapdragon/qwen3-tts-spec-experiment"
src=/tmp/llama-tts-spec-research
data=/tmp/qwen3tts-offset-run
run=/tmp/qwen3tts-grouped-run
model=/tmp/qwen3tts-spec-models/mmproj-qwen3-tts-0.6b-f16.gguf
reg="$HOME/Desktop/qwen3-tts-spec-raw-20261122/frames.bin"
mkdir -p "$run"
bash "$exp/build-replay.sh" "$src" "$src/build-host"
replay="$src/build-host/bin/qwen3tts-spec-replay"

TTS_SPEC_FLASH=1 TTS_SPEC_GROUPED=teacher "$replay" "$model" \
    "$data/offset-frames.bin" "$run/teacher.scores" 4 1 -1 >"$run/teacher.csv" 2>"$run/teacher.log"
OPENBLAS_NUM_THREADS=4 python3 "$exp/grouped.py" "$data" "$data/offset.scores" "$data/heads.bin" \
    "$run/teacher.scores" "$run/results" --regression "$reg"
TTS_SPEC_FLASH=1 TTS_SPEC_GROUPED="$run/results/early-bias.bin" "$replay" "$model" \
    "$data/offset-frames.bin" "$run/student.scores" 4 1 -1 >"$run/student.csv" 2>"$run/student.log"
OPENBLAS_NUM_THREADS=4 python3 "$exp/grouped.py" "$data" "$data/offset.scores" "$data/heads.bin" \
    "$run/teacher.scores" "$run/results" --student "$run/student.scores" --regression "$reg" >"$run/summary.txt"

cat "$data"/ev{01,02,03,04}.bin >"$run/eval.bin"
TTS_SPEC_FLASH=1 TTS_SPEC_GROUPED=ar-greedy "$replay" "$model" \
    "$run/eval.bin" "$run/eval-ar-greedy.scores" 4 1 -1
python3 "$exp/grouped-control.py" "$run" "$data" --prepare
TTS_SPEC_FLASH=1 TTS_SPEC_GROUPED=teacher "$replay" "$model" \
    "$run/eval-greedy-prefix.bin" "$run/eval-greedy-prefix.scores" 4 1 -1
python3 "$exp/grouped-control.py" "$run" "$data"
```

```sh
sdk="$HOME/Library/Android/sdk"
ndk_host="$sdk/ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64"
adb="$sdk/platform-tools/adb"
device=/data/local/tmp/q3tts-spec-20261122
device_mm=/data/local/tmp/q3tts/mmproj-qwen3-tts-0.6b-f16.gguf
CXX="$ndk_host/bin/aarch64-linux-android28-clang++" \
    bash "$exp/build-replay.sh" "$src" "$src/build-phone"
"$adb" push "$src/build-phone/bin/qwen3tts-spec-replay" "$device/qwen3tts-spec-grouped"
"$adb" push "$run/results/early-bias.bin" "$device/early-bias.bin"
"$adb" shell "cd $device && chmod 755 qwen3tts-spec-grouped"
for mode in ar-greedy early-bias.bin; do
    "$adb" shell "cd $device && LD_LIBRARY_PATH=. TTS_SPEC_PIN_MASK=f0 TTS_SPEC_FLASH=1 \
        TTS_SPEC_GROUPED=$mode taskset f0 ./qwen3tts-spec-grouped $device_mm frame0.bin group-$mode.scores \
        4 30 0 >group-$mode.csv 2>group-$mode.log"
    "$adb" pull "$device/group-$mode.csv" "$device/group-$mode.log" "$device/group-$mode.scores" "$run/"
done
```

[逐深度结果](results/grouped-depths.csv) · [greedy 对照](results/grouped-prefix-control.txt) · [AR 计时](results/grouped-phone-ar.csv) · [两阶段计时](results/grouped-phone-two-stage.csv) · [复跑](results/grouped-repeat.txt) · [散列](results/grouped.sha256)

原始捕获的 logits、安卓日志和训练产物在本机 `~/Desktop/qwen3-tts-spec-raw-20261122/grouped/`。
