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

## 第二批：改动、提交与结论（汇总）

这一节把本批的代码改动、提交、结论和复现方式集中在一处；细节与逐条数据在
[BENCHMARK.md 第二批各节](BENCHMARK.md)里。分支 `codex/qwen3-asr-batch`，**仅本地提交**，未推送。

### 代码改动

| 文件 | 改动 |
|---|---|
| `main.cpp` | 新增调度策略（`blocking` / `decode-first` / `interleaved` / `idle`，第 9–10 个参数）、`QWEN_FLASH_ATTN`、`QWEN_THREADS`、`QWEN_TIMING`；把 prefill 拆成可计时的 log-Mel / 音频 encoder / decoder prefill 三段 |
| `run_device.sh` | 新增 `QWEN_SCHEDULE`、`QWEN_BUDGET_MS`、`QWEN_EXTRA_ENV` 透传，文件名带上策略维度 |
| `compare_schedules.py` | 新增：读 `OUT/<schedule>/<trace>.<cap>.<round>.jsonl`，输出性能表、分阶段表和 token 一致性 |
| `ggml-hexagon.cpp` | 新增 `GGML_HEXAGON_TIMING`，分别打印每次图的 enqueue 与 sync 时间 |
| `BENCHMARK.md` | 新增第二批各节：分阶段、四档调度、量化与剖析、静态图否定、单请求收尾、优化后调度重跑、切块实验 |
| `README.md` | 新增调度策略说明与使用方式 |

### 提交

| 提交 | 内容 |
|---|---|
| `3e325aada` | 调度策略 + 分阶段计时 + `compare_schedules.py` + 运行器参数 |
| `065d1d291` | 单请求量化对照（Q8_0 / Q5_K / Q4_K / packed）与 HTP 算子剖析 |
| `89f285e98` | 更正 `SET_ROWS` 的判读（KV cache 写入，不是音频 embedding），按图拆派发成本 |
| `befa57e15` | 排除 `GGML_HEXAGON_OPBATCH` 512/2048/8192（无差别），新增 `QWEN_EXTRA_ENV` |
| `5de985dc5` | flash attention 改为默认（−16%）、编码器剖析（三层 conv 654 ms）、`GGML_HEXAGON_TIMING` |
| `7414563e0` | 否定静态图与编码器补齐（线性成本），记录 K/V 写随 flash 消失、Q4_K 精度一致 |
| `09874d5d7` | 单请求收尾：最终配置与“无效方向”清单 |
| `29080770f` | 优化后模型上的调度重跑 + 线程复核（2/4/8 无差别） |
| `1eeaa2a5e` | 1 s 编码块实测（慢 2 倍）与精度打断 |

### 结论

**单请求最优**：Q4_K + flash attention + 4 线程，一条 2.65 s 中文命令 **2348 ms**（基线 2980 ms，−21%），
分阶段 encoder 779 / decoder prefill 896 / decode 581 ms。

**有效的优化**：

| 手段 | 收益 |
|---|---|
| flash attention（本芯片上是正收益，与第一批假设相反） | prefill 1043→761 ms，单请求 −479 ms |
| Q4_K（q8 源再量化，zh/en/de 三条夹具文本一致） | decode 1024→744 ms |
| packed 打包（flash 关时有意义） | prefill −74 ms |

**无效或负收益（已记录，避免重复）**：`GGML_HEXAGON_OPBATCH` 三档、宿主线程 2/4/8、conv 图变体
（weight-first / gather）、静态图与编码器补齐（成本线性，补齐负收益）、1 s 编码切块（每块约 228 ms
固定开销，合计慢 2 倍且打断注意力）。

**调度**（同一 trace，capacity 2，capacity 1 做 token 对照）：`idle` 把首条终稿从 4124/4122/8633 ms
降到 2352/2340/7185 ms（约 −43%），代价是总耗时 +11%、配对次数归零；`blocking` 保留作为吞吐档。

**剩余瓶颈（应用层无法解决）**：encoder 的三层 conv2d（654 ms，`hvx-flat`、f16×f16→f32、2 MB VTCM，
约 44 GFLOP/s），以及 decode 每 token 的 DSP 执行（Q4_K 后约 58 ms）。前者要 DSP 侧融合 conv 核或
conv 权重 q8，后者只能靠 speculative decoding 或多请求合并提交。

### 复现

```sh
# 构建（宿主 + 设备）
cmake --build build-qwen-host --target llama-qwen3-asr-batch -j 8
docker run --rm --volume "$PWD:/workspace" --workdir /workspace --platform linux/amd64 -u $(id -u):$(id -g) \
  ghcr.io/snapdragon-toolchain/arm64-android:v0.8 \
  bash -c 'cmake --build /workspace/build-android-xasr --target llama-qwen3-asr-batch -j 6'

# 单次运行（换 QWEN_SCHEDULE 即换策略）
ADB=$ANDROID_SDK/platform-tools/adb QWEN_EXTRA_ENV='QWEN_FLASH_ATTN=1' QWEN_SCHEDULE=idle \
  examples/qwen3-asr-0.6b/run_device.sh build-android-xasr \
    "$CACHE/Qwen3-ASR-0.6B-q4k.gguf" "$CACHE/mmproj-Qwen3-ASR-0.6B-Q8_0-convout-q8.gguf" \
    "$NDK/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" \
    /tmp/qwen-solo /tmp/qwen-q/final 1

# 汇总
python3 examples/qwen3-asr-0.6b/compare_schedules.py --dir /tmp/qwen-b2
```

夹具：`simultaneous` / `staggered` / `refill` 三份 trace，`zh` / `en` / `de` / `long` 四段音频（本地 TTS 生成，
`fixtures.sha256` 记录哈希）；归档 `~/Library/Caches/qwen3-asr-htp/qwen3-asr-0.6b-batch2-schedule-sm6650-2026-11-19.tar.gz`
（SHA-256 `25ef540d…`）。

### 未决与边界

- **F16 源重量化未做**：本机只有 Q8_0 GGUF，Q4_K 是从 Q8_0 二次量化；要真正的 F16→Q4_K 需要原始 checkpoint。
- **样本量**：调度 2 轮、其余 1 轮，P95/P99 的样本数只有 6–12；结论方向可靠，绝对数字会有几 % 浮动。
- **工作区仍有他人未提交改动**：`conversion/qwen3tts.py`、`examples/qwen3-asr-0.6b/compare.py`，本批未触碰。
