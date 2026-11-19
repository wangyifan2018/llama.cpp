# Qwen3-ASR-0.6B 两路 continuous batching

这个 example 重放整段音频按时间进入的请求队列。它复用 MTMD 音频预处理、Qwen3-ASR 投影器、模型模板和 llama.cpp KV cache。trace 的到达时间表示完整语段可提交的时刻。语言由模型判断，JSONL 保留原始模型文本、清理后的终稿和 token。

每个槽位有独立的 KV sequence ID 和位置。每轮最多接收一个已经到达的请求，完成 MTMD prefill，然后在一次 decoder 调用里为所有活跃槽位提交各一个 token。请求结束时释放该 sequence 的 KV，下一条已到达的请求可补位。单路可以直接推进。音频 encoder 与 prompt prefill 当前逐条执行。双路档的验收以 token 一致、实际配对执行、错峰补位以及同设备上较短的总耗时为准。单请求延迟单独列出。

容量 1 和容量 2 使用同一个二进制、权重、4 个 CPU 线程、greedy sampler、2048 总上下文和 2 个内部 KV sequence。两档先用同一真实语段预热再开始计时。HTP 档关闭 Flash Attention，并显式选择 HTP 做 decoder 层及音频 projector。实际算子归属以设备日志为准。

## 构建

```sh
cmake -S . -B build-qwen-host -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_MTMD=ON -DLLAMA_BUILD_TOOLS=OFF
cmake --build build-qwen-host --target llama-qwen3-asr-batch -j 8
```

Android 用仓库的 Snapdragon 构建预设和 v0.8 镜像，NDK r29、Hexagon SDK 6.7.0.0。 [Snapdragon 编译说明](../../docs/backend/snapdragon/README.md)

```sh
test -f CMakeUserPresets.json || cp docs/backend/snapdragon/CMakeUserPresets.json .
QA_IMAGE=ghcr.io/snapdragon-toolchain/arm64-android@sha256:e39e5e9c0957550a92c0de756b7556d81196c7c7f7328ed80de252bfb4878adf
docker run --rm --platform linux/amd64 -v "$PWD:/workspace" -w /workspace \
    -u "$(id -u):$(id -g)" "$QA_IMAGE" sh -ec '
    cmake --preset arm64-android-snapdragon-release -B build-qwen-android \
        -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_UI=OFF
    cmake --build build-qwen-android --target llama-qwen3-asr-batch -j 8
'
```

## 准备权重和音频

开发机缓存为 `$HOME/Library/Caches/qwen3-asr-htp`。模型为 Q8_0，projector 将 `a.conv_out.weight` 从 F16 重新量化为 Q8_0，两档保持相同权重。 [Qwen3-ASR 模型](https://github.com/QwenLM/Qwen3-ASR) · [GGUF 权重](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF)

```sh
QA_EX="$PWD/examples/qwen3-asr-0.6b"
QA_MODELS="$HOME/Library/Caches/qwen3-asr-htp"
QA_OUT=/tmp/qwen-batch2
mkdir -p "$QA_OUT"
hf download ggml-org/Qwen3-ASR-0.6B-GGUF Qwen3-ASR-0.6B-Q8_0.gguf \
    mmproj-Qwen3-ASR-0.6B-Q8_0.gguf --local-dir "$QA_MODELS" \
    --revision 928ab958557df9aa2ef1c93e0e83c7ad0933fae2
docker run --rm --platform linux/amd64 "$QA_IMAGE" sh -ec '
    cat "$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
' >"$QA_OUT/libc++_shared.so"
PYTHONPATH="$PWD/gguf-py" python3 "$QA_EX/repack_projector.py" \
    "$QA_MODELS/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf" "$QA_OUT/projector.gguf"
python3 "$QA_EX/make_traces.py" "$QA_MODELS" "$QA_OUT/fixtures"
(cd "$QA_OUT/fixtures" && sha256sum -c "$QA_EX/fixtures.sha256")
```

| 文件 | SHA-256 |
|---|---|
| `Qwen3-ASR-0.6B-Q8_0.gguf` | `bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971` |
| `mmproj-Qwen3-ASR-0.6B-Q8_0.gguf` | `41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d` |
| `projector.gguf` | `19197c150ceba787046dcec9dacfc302522383a97fac9fe18bc57a9593e5280f` |
| `libc++_shared.so` | `0c52cfab2df0d957d8b346a2bdc5ae8d71feca2591924d77e1cd724d5bf74352` |

三条基础 WAV 是 16 kHz 单声道 PCM16，分别为中文导航、英文空调和德语导航。8.32 s 长片段由中文短音频拼接三次。trace 每行是到达时间（毫秒）、Tab、文件路径。主机使用同一套权重和 trace，命令为 `llama-qwen3-asr-batch MODEL MMPROJ CPU SLOTS TRACE.tsv`，从夹具目录启动。

## 真机复现

设置 `ADB` 为本机 Android SDK 的 adb 路径。多台设备连接时设置 `ANDROID_SERIAL`。runner 需要交叉编译 NDK 的 ARM64 `libc++_shared.so`。设备 runner 使用 `/data/local/tmp` 下的独立目录，校验传输哈希，连续执行串行和双路。运行目录保存日志、每条请求、配对 decode 和校验结果。

```sh
QWEN_PROFILE=1 "$QA_EX/run_device.sh" "$PWD/build-qwen-android" \
    "$QA_MODELS/Qwen3-ASR-0.6B-Q8_0.gguf" "$QA_OUT/projector.gguf" \
    "$QA_OUT/libc++_shared.so" "$QA_OUT/fixtures" "$QA_OUT/device" 3
python3 "$QA_EX/compare.py" "$QA_OUT/device" 3 >"$QA_OUT/device/summary.json"
```

基准覆盖同时到达、300 ms 错峰和第三条等待后补位。比较器检查 slot 生命周期、两路真实 decode 调用、第三条确实与仍活跃的序列组成 batch，以及每个请求的输出 token 一致性。按到达至终稿计算延迟和 P50/P95/P99。总时长从 trace 开始至全部完成，吞吐为请求数除以总时长的中位数。每轮用新进程和同一预热语段，单双路执行顺序交替。profiling 单独执行。

[SM6650 实测记录](BENCHMARK.md)包含串行与双路的 A/B 结果、调度与 HTP 算子证据，以及可解包复测的夹具和原始日志位置。

## 调度策略（第二批）

第 9、10 个参数选择“两次 decode 之间算谁的”：

```text
llama-qwen3-asr-batch MODEL MMPROJ CPU|HTP0:0 SLOTS TRACE.tsv \
    [MAX_TOKENS [eager|windowed|packed [blocking|decode-first|interleaved|idle [PREFILL_BUDGET_MS]]]]
```

| 策略 | 行为 |
|---|---|
| `blocking` | 默认，第一批的行为：新请求的整段 prefill 插在活跃序列两次 decode 之间 |
| `decode-first` | 先 decode 再 prefill，prefill 仍整段 |
| `interleaved` | 每轮只推进一个 prefill chunk（`PREFILL_BUDGET_MS` 为毫秒预算，-1 表示一个 chunk） |
| `idle` | 有活跃序列时不启动新 prefill，只在其结束后开始 |

实测（SM6650，2 slot，2–3 条请求）：`idle` 把首条终稿从 5.07 s 降到 2.94 s（−41%），代价是总耗时
+14%、P95 变差、配对次数归零；`interleaved` 只省 70–80 ms，因为 790 ms 的音频 encoder 在 mtmd 里
仍是一块。完整对照与分阶段分解见 [BENCHMARK.md](BENCHMARK.md#第二批调度策略与首个终稿延迟2026-11-19)。
