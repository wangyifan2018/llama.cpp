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

## 4. 日常增量迭代

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
