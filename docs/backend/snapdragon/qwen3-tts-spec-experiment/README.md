# E28 Qwen3-TTS CodePredictor 窗口 2 回放实验

2026-11-22。在现有 Qwen3-TTS 12 Hz 0.6B Base 上测未来 RVQ head 的草稿质量和两位置验证成本。基础提交为 `b9d75b1fbfad0868cf471a4ac0569f8ba78d9d28`。实验改动保存在 [research.patch](research.patch)，仅应用到独立研究 worktree。

[E29 共享偏置蒸馏](offset-distillation.md) 使用本实验的 AR 帧作回归集。

## 怎么测

- 用原 AR 路径生成四段各 24 帧的语音，中英文各一条短文本，每条用两个 seed。同一份参考音色。采样设置为 top-k 50、top-p 0.8、temperature 1。两台机器消费同一份 96 帧输入，安卓单核 Flash Attention 对照只回放首帧。
- 每帧保留 Talker hidden 和 16 个已生成 RVQ code。以真实 code 前缀回放，逐深度计算目标 logits p 和用上一位置 hidden 乘当前 head 得到的草稿 logits q。先对两者应用同一套过滤，再用 `sum(min(p, q))` 计算该前缀下的理论接受率。另算 draft top-1 命中率，和单步/两位置验证的 argmax 一致率。
- 单轮计时固定第一条英文样本的第一帧，分别在待生成 `code2`、`code8`、`code14` 时复用精确的 AR 前缀 KV。T1 执行单位置主干和一个目标 head，T2 执行两位置主干、两个目标 head 和一个草稿 head。两位置带因果 mask。主试验使用与原 AR 路径匹配的 CPU Flash Attention，首轮普通 attention 数据作为对照。每组固定输入预热 3 次，`AB/BA/AB/BA` 四块各 20 次，得到每档 80 个时间。计时包含 CPU backend graph compute 内部的计划开销，图的搭建、分配和输入上传在计时区间外。
- 验证图使用已知 AR code 填入草稿位置。计时覆盖主干和 head，实际采样、草稿依赖的运行时编排、拒绝回退、KV 提交、跨设备同步和音频端到端耗时留给完整解码实验。表里的 `(1+a)/(T2/T1)` 只估算可产出 bonus code 的单轮，其中 a 是该深度的平均接受率。

## 实测结果

| 待预测深度 | 平均接受率 | Mac 4 线程 T1 / T2 | 安卓单核 T1 / T2 | 安卓逐线程绑核 T1 / T2 |
|---|---:|---:|---:|---:|
| code2 | 2.42% | 3.107 / 3.921 ms | 20.371 / 39.976 ms | 11.906 / 15.320 ms |
| code8 | 47.42% | 3.116 / 3.974 ms | 20.440 / 40.109 ms | 11.893 / 15.368 ms |
| code14 | 53.15% | 3.192 / 4.085 ms | 20.579 / 40.276 ms | 11.918 / 15.471 ms |

- 深度 2–14 的平均接受率为 **41.23%**。四条样本分别为 43.28%、40.08%、41.72%、39.84%（Mac）。同一批前缀只用 top-k 50 时，平均接受率为 44.07%。两组都表现为前几层低、中后段高。
- Flash Attention 下，Mac 和安卓四线程的单步、两位置验证 logits 在 1,440 个位置上各自逐元素相等。Mac 另对原 AR 路径前 12 帧、180 个位置做了采样前 logits 对照，与单步回放逐元素相等。重建的 top-k/top-p 支持集覆盖原 AR 采出的 1,440 个 code。普通 attention 对照出现三个 cutoff 附近的不一致，抽查确认了其中一个由 attention 路径差异导致。Mac 原整帧 predictor 日志去掉每段的第一帧后，中位数为 56.95 ms/帧。
- 安卓绑单核时，T2/T1 约 **1.96**，高过这组接受率能支撑的盈亏线。逐线程绑 CPU4–7 时，比值降至 **1.29–1.30**。深度 8 和 14 的算子级单轮估算分别为 **1.14×** 和 **1.18×**，深度 2 约 **0.80×**。深度 8 和 14 留给额外运行时开销的余量分别约 **2.16 和 2.78 ms/轮**，按 `T1 × (1+a) − T2` 计算。对应的 Mac 算子级估算为 1.16×、1.20×、0.81×。
- 安卓最初只用 `taskset f0` 限制四线程的可运行核心，约一分钟后仍无落盘计时样本，随后终止该进程。给每个 worker 单独固定核心后，单帧试跑与 96 帧正式回放都完成。普通 attention 单帧试跑的 T2/T1 为 1.28–1.31。调度差异的原因没有单独定位。

**迁移时先保留深度 1–4 的 AR 基线，优先测深度 8–14。** 在目标后端记录热态 T2/T1，再检查是否低于各深度的 `1+a`。当比值贴近门槛时，先测批量矩阵乘、线程放置和提交成本。算子级通过后，再对比现有整帧 AR 图的 predictor ms/帧，额外记录冷启动与端到端 RTF。当前结果覆盖同一个 0.6B checkpoint、同一音色、两个短文本，设备换代或换量化档后复测。

详细记录见 [Mac](results/mac-flash-summary.txt)、[安卓逐线程绑核](results/android-four-flash-summary.txt)、[安卓单核计时](results/android-single-flash-timing.csv)和 [top-p=1](results/mac-flash-top-p1-summary.txt)。`results/` 还包含逐深度、逐帧指标，普通 attention 对照、计时 CSV 和模型/输入散列。原始二进制、采样对照补丁与 CLI 日志在本机 `~/Desktop/qwen3-tts-spec-raw-20261122/`。

## 复跑

下面的路径可换成目标机器上的位置。Android 使用 Android NDK r28c，Apple 使用 Apple clang 21.0.0。两个构建都用 Release、CPU、BLAS 和 OpenMP 关闭，Android 另外启用 `armv8.2-a+dotprod+fp16`。采用共享库默认设置，Mac 的 GGML_ACCELERATE 保持默认开启。Mac 为 MacBook Air M2、8 GB、macOS 26.5.2，实验用 4 线程且未绑核。安卓为 MTN-AN80、SM6650、Android 16。

在 llama.cpp 根目录执行：

```sh
exp="$PWD/docs/backend/snapdragon/qwen3-tts-spec-experiment"
src=/tmp/llama-tts-spec-research
models=/tmp/qwen3tts-spec-models
run=/tmp/qwen3tts-spec-run
base=b9d75b1fbfad0868cf471a4ac0569f8ba78d9d28
git worktree add --detach "$src" "$base"
git -C "$src" apply "$exp/research.patch"

cmake -S "$src" -B "$src/build-host" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
cmake --build "$src/build-host" --target llama-tts -j 7
bash "$exp/build-replay.sh" "$src" "$src/build-host"
```

模型和参考文件取自旧实验设备，文件名与散列见 [inputs.sha256](results/inputs.sha256)。现有设备路径为 `/data/local/tmp/q3tts/`，可用 `adb pull` 取回这三个文件。已持有原始回放输入时，可以直接使用 `frames.bin`，省去第一条生成命令。同一台 Mac 重做采集后，四段输入及合并文件的 SHA-256 都一致，见 [复跑记录](results/capture-repeat-check.txt)。另一台机器重新生成时，浮点舍入和采样可能使散列变化，跨后端比较应使用同一个文件。

```sh
(cd "$models" && shasum -a 256 -c "$exp/results/inputs.sha256")
bash "$exp/capture.sh" "$src/build-host/bin/llama-tts" "$models" "$run"
TTS_SPEC_FLASH=1 "$src/build-host/bin/qwen3tts-spec-replay" \
    "$models/mmproj-qwen3-tts-0.6b-f16.gguf" "$run/frames.bin" "$run/mac-flash.scores" \
    4 20 0 >"$run/mac-flash-timing.csv" 2>"$run/mac-flash-replay.log"
python3 "$exp/analyze.py" "$run/mac-flash.scores" "$run/mac-flash-timing.csv" \
    "$run/mac-flash-per-frame.csv" --capture "$run/frames.bin"
```

Android 构建与独立目录部署如下。换设备时，调整 NDK host tag、模型路径及 CPU mask。SM6650 上 `0x80` 是单独的大核 CPU7，`0xf0` 是 CPU4–7。四线程版本同时指定进程可运行的核心和每个 worker 的固定核心。

```sh
sdk="$HOME/Library/Android/sdk"
ndk="$sdk/ndk/28.2.13676358"
ndk_host="$ndk/toolchains/llvm/prebuilt/darwin-x86_64"
adb="$sdk/platform-tools/adb"
device=/data/local/tmp/q3tts-spec-20261122
device_mm=/data/local/tmp/q3tts/mmproj-qwen3-tts-0.6b-f16.gguf

cmake -S "$src" -B "$src/build-phone" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_shared \
    -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_NATIVE=OFF \
    -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
cmake --build "$src/build-phone" --target mtmd -j 4
CXX="$ndk_host/bin/aarch64-linux-android28-clang++" \
    bash "$exp/build-replay.sh" "$src" "$src/build-phone"

phone_bin="$src/build-phone/bin"
"$adb" shell "mkdir -p $device"
"$adb" push "$phone_bin/"*.so "$phone_bin/qwen3tts-spec-replay" \
    "$ndk_host/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" "$run/frames.bin" "$device/"
"$adb" shell "cd $device && chmod 755 qwen3tts-spec-replay"

"$adb" shell "cd $device && LD_LIBRARY_PATH=. TTS_SPEC_PIN_MASK=f0 TTS_SPEC_FLASH=1 taskset f0 \
    ./qwen3tts-spec-replay $device_mm frames.bin android-four-flash.scores 4 20 0 \
    >android-four-flash-timing.csv 2>android-four-flash-replay.log"
"$adb" shell "cd $device && head -c 4160 frames.bin > frame0.bin && LD_LIBRARY_PATH=. TTS_SPEC_FLASH=1 \
    taskset 80 ./qwen3tts-spec-replay $device_mm frame0.bin android-single-flash.scores 1 20 0 \
    >android-single-flash-timing.csv 2>android-single-flash-replay.log"

head -c 4160 "$run/frames.bin" >"$run/frame0.bin"
for variant in android-four-flash android-single-flash; do
    capture="$run/frames.bin"
    if [ "$variant" = android-single-flash ]; then capture="$run/frame0.bin"; fi
    "$adb" pull "$device/$variant.scores" "$device/$variant-timing.csv" "$device/$variant-replay.log" "$run/"
    python3 "$exp/analyze.py" "$run/$variant.scores" "$run/$variant-timing.csv" "$run/$variant-per-frame.csv" --capture "$capture"
done
```

去掉 `TTS_SPEC_FLASH=1` 可复跑普通 attention 对照。对已有 logits 运行 `analyze.py --top-p 1` 可重做只用 top-k 的对照。设备通过 USB 调用 adb，计时未锁定频率或记录温度。

`frames.bin` 是 little-endian 固定长度记录。本模型每帧由 1024 个 float32 hidden 元素和 16 个 int32 code 组成，共 4160 字节。输出 `.scores` 以两个 int32 开头，分别是 RVQ 预测深度数和词表大小。每帧随后是按深度排列的单步目标 logits、草稿 logits（从深度 2 起）、宽验证目标 logits。时间 CSV 的 `repeat` 可以用来还原每个交替计时块。
