# X-ASR streaming on GGML and Hexagon HTP

This example runs the released [X-ASR-zh-en](https://github.com/Gilgamesh-J/X-ASR) Zipformer2 transducer for Mandarin and English. It uses the 480 ms streaming configuration and emits one final transcript per utterance. The inference runtime is C++ and builds against this repository's GGML. The graph and the streaming decoder are adapted from [CrispASR at 9ddae459](https://github.com/CrispStrobe/CrispASR/tree/9ddae4590d427f27e0ca04694864eed66bcb530e) under [its MIT license](LICENSE.crispasr).

See [BUILD.md](BUILD.md) for host compilation, the full Android HTP build, device deployment, and incremental rebuilds.

## Build and run on CPU

From the repository root:

```sh
cmake -S . -B build-x-asr -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build-x-asr --target llama-x-asr -j
curl -fL -o x-asr-zh-en-q8_0.gguf \
    https://huggingface.co/cstr/x-asr-zh-en-GGUF/resolve/main/x-asr-zh-en-q8_0.gguf

build-x-asr/bin/llama-x-asr x-asr-zh-en-q8_0.gguf CPU zh.wav en.wav
build-x-asr/bin/llama-x-asr x-asr-zh-en-q8_0.gguf CPU zh.wav en.wav --realtime
```

Use the [published Q8_0 GGUF](https://huggingface.co/cstr/x-asr-zh-en-GGUF) for the measured configuration. The CLI uses miniaudio to decode and resample to mono 16 kHz. It delivers audio to the runtime in 20 ms packets. During unpaced execution it reports wall-clock real-time factor. With `--realtime`, it reports elapsed time from the scheduled arrival of the last packet to the final text, including pending inference and flush. Both modes report `flush_ms` measured from the actual call to commit.

## Build and run on Android HTP

Use the repository's [Snapdragon build and deployment instructions](../../docs/backend/snapdragon/README.md). With the Android toolchain and ADB set up, the existing build helper installs the example along with the matching ARM64 and DSP libraries.

```sh
python3 scripts/snapdragon/build.py --target android --push
adb push x-asr-zh-en-q8_0.gguf zh.wav en.wav /data/local/tmp/llama.cpp/
adb shell 'cd /data/local/tmp/llama.cpp &&
    export LD_LIBRARY_PATH="$PWD/lib" ADSP_LIBRARY_PATH="$PWD/lib" &&
    export GGML_HEXAGON_DEVICES=HTP0:0 &&
    bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 zh.wav en.wav --realtime'
```

Set the positional device to `CPU` for a same-device comparison. For the measured low-latency SM6650 configuration, export `GGML_HEXAGON_OPPOLL=1` and `GGML_HEXAGON_OPBATCH=2048` before launching the binary. This example checks the selected device's actual scheduler placement and fails initialization if any encoder compute node falls back. The placement line includes the device, assigned nodes, scheduler splits, and bytes of encoder cache. One scheduler split can still produce multiple HTP command batches.

## Inference flow

| Stage | Execution |
| --- | --- |
| PCM to 80-bin Kaldi fbank | CPU, incremental as packets arrive |
| Conv2d subsampling, ConvNeXt, six Zipformer stacks, encoder projection | Selected GGML device |
| Encoder state update | Same device, as copy nodes at the end of the encoder graph |
| Transducer predictor, joiner, greedy token selection and text formatting | CPU |

The encoder consumes 61 feature frames and advances by 48 frames. Each chunk yields 12 encoder frames. The graph is allocated once and reuses the same state tensors. Each chunk transfers its features and masks into the graph and reads the encoder projection back to the CPU. The 115 state tensors stay in the graph between chunks.

The graph uses sum-of-squares and contiguous square root for BiasNorm. It maps the ConvNeXt depthwise convolution to the native depthwise op. Fixed relative-position projections and convolution edge scales are computed once at graph initialization. These changes use the existing weights. Small floating-point differences are possible from reduction order and kernel precision.

For C++ integration, load the GGML backends, open a context, create a stream, and pass normalized 16 kHz mono PCM to `xasr_append`. Call `xasr_commit` at the upstream VAD endpoint and destroy the stream before creating the next one. The context accepts one active stream, with a ten-minute audio limit. Each append or commit call runs synchronously. The API is in [xasr.h](xasr.h).

The example covers the inference path. A server can put a request queue and IPC around this API. True in-flight batching will require independent persistent stream state and a batched encoder graph.

See [BENCHMARK.md](BENCHMARK.md) for the reproducible device benchmark, fixture hashes, and results. [OPTIMIZATION.md](OPTIMIZATION.md) explains the graph changes, how to isolate each one, and the next experiments.
