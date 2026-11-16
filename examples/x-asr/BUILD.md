# 编译和部署 X-ASR

所有宿主机命令都从 llama.cpp 仓库根目录执行。宿主机编译用于 CPU 验证。Android 包使用本仓库的 Snapdragon 工具链交叉编译，并在真机上分别验证 CPU 和 HTP。

## 1. 宿主机 CPU 编译

需要 CMake 和 C/C++ 编译器。下面只构建 `llama-x-asr` 及其依赖，独立使用 `build-x-asr-host` 目录。

```sh
cmake -S . -B build-x-asr-host -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DGGML_HEXAGON=OFF
cmake --build build-x-asr-host --target llama-x-asr --parallel
build-x-asr-host/bin/llama-x-asr x-asr-zh-en-q8_0.gguf CPU zh.wav en.wav
```

模型下载、音频来源和 SHA-256 见 [BENCHMARK.md](BENCHMARK.md#reproduce-on-android)。把模型和 WAV 放到仓库根目录，或在命令中传入实际路径。CLI 会把音频转换为 16 kHz 单声道 PCM，并按 20 ms 分包。加 `--realtime` 测最后一包计划到达至 final 的延迟，省略该参数测不等待音频时钟的 RTF。

## 2. Android ARM64 + HTP 全量构建与部署

宿主机需要 Python 3、运行中的 Docker 和 Android Platform Tools。ADB 在宿主机运行，交叉编译在 Snapdragon Docker 镜像中运行。SDK 和工具链配置见 [Snapdragon 指南](../../docs/backend/snapdragon/README.md)。

```sh
export PATH="$HOME/Library/Android/sdk/platform-tools:$PATH" # macOS 默认 SDK 路径
export ANDROID_SERIAL=ADNQ6R5823007930
adb devices
adb -s "$ANDROID_SERIAL" shell 'getprop ro.soc.model; getprop ro.product.cpu.abi'

python3 scripts/snapdragon/build.py \
    --target "android:$ANDROID_SERIAL" \
    --build-dir build-android-xasr --install-dir pkg-android-xasr \
    --toolchain-version v0.8 \
    --target-dir /data/local/tmp/llama-x-asr --push
```

脚本使用 Android Snapdragon Release preset，编译、安装并推送整个包。当前设备应报告 `SM6650` 和 `arm64-v8a`。产物在宿主机的 `build-android-xasr/bin/llama-x-asr` 和 `pkg-android-xasr/llama.cpp/`。设备目录中应有：

| 路径 | 用途 |
|---|---|
| `bin/llama-x-asr` | X-ASR 可执行文件 |
| `lib/libggml*.so` | Android ARM64 GGML 运行时、CPU 后端和 Hexagon 宿主库 |
| `lib/libggml-htp-v73.so` | SM6650 上在 DSP 端加载的 v73 库 |
| `lib/libc++_shared.so` | 二进制使用共享 C++ 运行时时需要的 Android NDK 库 |

如需延后部署，首次调用省略 `--push`，之后从宿主机推送已安装的包：

```sh
adb -s "$ANDROID_SERIAL" shell 'mkdir -p /data/local/tmp/llama-x-asr'
adb -s "$ANDROID_SERIAL" push pkg-android-xasr/llama.cpp/. /data/local/tmp/llama-x-asr/
adb -s "$ANDROID_SERIAL" shell 'chmod 755 /data/local/tmp/llama-x-asr/bin/llama-x-asr'
```

## 3. 推送模型并验证 CPU 和 HTP

```sh
adb -s "$ANDROID_SERIAL" push x-asr-zh-en-q8_0.gguf zh.wav en.wav /data/local/tmp/llama-x-asr/

adb -s "$ANDROID_SERIAL" shell -T 'cd /data/local/tmp/llama-x-asr &&
    export LD_LIBRARY_PATH="$PWD/lib" ADSP_LIBRARY_PATH="$PWD/lib" &&
    bin/llama-x-asr x-asr-zh-en-q8_0.gguf CPU zh.wav en.wav --realtime'

adb -s "$ANDROID_SERIAL" shell -T 'cd /data/local/tmp/llama-x-asr &&
    export LD_LIBRARY_PATH="$PWD/lib" ADSP_LIBRARY_PATH="$PWD/lib" &&
    export GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_OPPOLL=1 &&
    bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 zh.wav en.wav --realtime'
```

`LD_LIBRARY_PATH` 用于 Android 进程加载共享库，`ADSP_LIBRARY_PATH` 用于 DSP 加载 v73 库。当前图的 HTP 验收日志应包含 `encoder HTP0:0 2877/2877 ops, 1 scheduler splits`。优化图后总节点数可能变化，检查已分配数等于总数，并和 CPU 对照转写。若找不到 `HTP0:0`，检查设备架构、库目录和 Hexagon 会话初始化日志。若出现 `CANNOT LINK EXECUTABLE`，先核对 Android ABI、报错中的共享库和构建包。若缺少 `libc++_shared.so`，使用同一构建 NDK 的 arm64 版本。

## 4. 部署 ASR 服务端

服务端是独立目标 `llama-x-asr-server`，与 CLI 共用同一份引擎。宿主机验证：

```sh
cmake --build build-x-asr --target llama-x-asr-server -j
build-x-asr/bin/llama-x-asr-server x-asr-zh-en-q8_0.gguf CPU --port 8765 --streams 2
```

Android 上运行（HTP 引擎 2 路 + CPU 引擎 2 路，HTP 参数与 CLI 相同）：

```sh
adb -s "$ANDROID_SERIAL" shell -T 'cd /data/local/tmp/llama-x-asr &&
    export LD_LIBRARY_PATH="$PWD/lib" ADSP_LIBRARY_PATH="$PWD/lib" &&
    export GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_OPPOLL=1 &&
    nohup bin/llama-x-asr-server x-asr-zh-en-q8_0.gguf HTP0:0 \
        --port 8765 --streams 2 --cpu-streams 2 --idle-ms 30000 \
        > server.log 2>&1'

adb -s "$ANDROID_SERIAL" forward tcp:8765 tcp:8765
python3 examples/x-asr/tools/asr_client.py --port 8765 --wav zh.wav --sessions 4 --paced
```

启动日志应包含每个引擎一行 `engine HTP0:0 with 2 streams` / `engine CPU with 2 streams`，以及 `idle timeout 30000 ms, listening on 0.0.0.0:8765`。客户端日志逐路给出 `endpoint_to_final_ms` 与转写。

协议、队列与背压、超时语义见 [README.md](README.md#asr-server)；端到端与分位数见 [BENCHMARK.md](BENCHMARK.md#server-end-to-end)。端到端通路自检：

```sh
MODEL=x-asr-zh-en-q8_0.gguf examples/x-asr/tools/server_smoke.sh "zh.wav en.wav" CPU
```

识别质量自检（宿主机 TTS 合成 24 条带标注命令，输出 CER/WER）：

```sh
examples/x-asr/tools/make_eval_pack.sh /tmp/eval-pack
sed 's/-snr10\.wav/.wav/' /tmp/eval-pack/manifest.tsv > /tmp/eval-pack/manifest-clean.tsv
python3 examples/x-asr/tools/eval_cer.py --manifest /tmp/eval-pack/manifest-clean.tsv \
    --model x-asr-zh-en-q8_0.gguf --device CPU --label "CPU default"
```

服务指标采样（固定到达序列、分位数）：

```sh
python3 examples/x-asr/tools/bench_service.py --port 8765 --reps 12 \
    --short zh.wav --long zh-10.wav --long2 en-20.wav --out /tmp/svc.jsonl
python3 examples/x-asr/tools/bench_service.py --port 8765 --throughput --concurrency 1,2,4 \
    --short zh.wav --long zh-10.wav --long2 en-20.wav
```

## 5. 日常增量迭代

首次完整部署后，只修改 `examples/x-asr` 时，可在同一镜像、同一容器路径下重用 CMake 构建目录，并只更新可执行文件。

```sh
docker run --rm --platform linux/amd64 \
    --user "$(id -u):$(id -g)" --mount "type=bind,source=$PWD,target=/workspace" \
    --workdir /workspace ghcr.io/snapdragon-toolchain/arm64-android:v0.8 \
    cmake --build /workspace/build-android-xasr --target llama-x-asr --parallel

adb -s "$ANDROID_SERIAL" push build-android-xasr/bin/llama-x-asr /data/local/tmp/llama-x-asr/bin/
adb -s "$ANDROID_SERIAL" shell 'chmod 755 /data/local/tmp/llama-x-asr/bin/llama-x-asr'
```

如果改了 GGML、Hexagon 宿主后端、DSP kernel 或工具链配置，重新走第 2 节以同步部署可执行文件和依赖库。做性能对照时保留构建版本和设备上的实际库版本。完整的尾延迟、RTF 和 HTP profile 复测命令见 [BENCHMARK.md](BENCHMARK.md)。
