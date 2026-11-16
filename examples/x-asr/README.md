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

## ASR server

`llama-x-asr-server` wraps the engine in a small daemon. It is not part of
`tools/server`: the [server scope](tools/server/README-dev.md) covers
model-agnostic LLM features and excludes model-specific ones, and this is a
model-specific streaming ASR engine with its own frame protocol.

```sh
cmake --build build-x-asr --target llama-x-asr-server -j
build-x-asr/bin/llama-x-asr-server x-asr-zh-en-q8_0.gguf CPU --port 8765 --streams 4
python3 examples/x-asr/tools/asr_client.py --port 8765 --wav zh.wav --sessions 4 --paced

# Device engine with two slots plus a CPU engine with two slots.
build-x-asr/bin/llama-x-asr-server x-asr-zh-en-q8_0.gguf HTP0:0 \
    --port 8765 --streams 2 --cpu-streams 2
```

`--cpu-streams N` starts a second engine on the CPU backend with its own worker
thread and slots. Sessions are pinned at OPEN to the least loaded engine that has
a free slot, so a burst of simultaneous endpoints is split across both
processors. It costs a second copy of the weights, about 200 MB, and halves the
four-way tail (see [BENCHMARK.md](BENCHMARK.md#hybrid-engines)).

On Android, run the daemon on the device and forward the port:

```sh
adb shell -T 'cd /data/local/tmp/llama.cpp &&
    export LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib &&
    export GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_OPPOLL=1 &&
    bin/llama-x-asr-server x-asr-zh-en-q8_0.gguf HTP0:0 --port 8765 --streams 4'
adb forward tcp:8765 tcp:8765
```

### Protocol

Length-prefixed frames, little-endian. Request: `u32 op, u32 session, u32 bytes,
payload`. Response: `u32 status, u32 bytes, payload`.

| op | name | payload | reply |
|---:|---|---|---|
| 0 | OPEN | none | status 0, 4-byte session id |
| 1 | APPEND | float32 mono 16 kHz | none, one way |
| 2 | COMMIT | none | status 1, UTF-8 final text |
| 3 | CANCEL | none | status 0, the stream is freed |
| 4 | CLOSE | none | status 0, the stream is freed |

Status 2 carries an error string. APPEND is one way: audio never waits for a
reply, and backpressure is the request queue filling and the reader blocking on
the socket. Every other op answers exactly once. An APPEND the engine rejects
closes the connection, so a client can never resynchronise on a stale reply.

The VAD stays on the caller's side: the client sends PCM while it records and
one COMMIT at its endpoint. The server returns only final text.

### Lifecycle, limits, scheduling

- One worker thread owns the engine and every session; connection threads only
  move bytes. The engine is not thread safe, so this is also what makes the
  context usable.
- `--streams N` bounds concurrent sessions, because a session holds one encoder
  slot. OPEN past the limit returns status 2 and the caller must wait or retry.
  CLOSE or CANCEL returns the slot. COMMIT finalizes the utterance. A client that stops sending is handled by idle expiry.
- `--queue N` bounds the pending request queue (256 by default). When it is full
  the reader thread blocks, which turns into TCP backpressure on the client.
- One request is held to 60 s of audio and one stream to ten minutes, and a
  session with no request for `--idle-ms` (30 s by default) is released and its
  slot returned. Expiry is queued to the owning engine, so a stream is never
  freed from the janitor thread while its worker is inside a commit.
- A connection that stops sending is dropped after 60 s without a complete
  frame, so a stalled client cannot pin a reader thread.
- COMMIT has priority: when one is queued, that session's earlier APPENDs are
  drained and run first, then the commit. Other sessions' audio cannot delay a
  finished utterance, and per-session order is preserved.
- `--chunk-scheduler` enables the experimental B=1 scheduler. An APPEND or
  COMMIT can yield after one encoder chunk and its greedy decode. The engine
  alternates new work with resumable requests while preserving the order of
  operations for a session. The default remains request-level scheduling.
  See the [scheduler A/B](BENCHMARK.md#b1-chunk-scheduler) for reproduction.
- A client that disconnects without CLOSE is reaped by the idle timeout.
- Degenerate input: an all-zero segment drives the encoder normalisation to zero
  variance and the logits to NaN. The decoder now clamps an out of range argmax
  to blank, so the process survives, but the transcript for such a segment is
  not meaningful: a 2 s silent clip returns empty text, while 20 ms of digital
  silence returned a stray token on one backend and nothing on the other. Real
  VAD segments carry noise; this is a guard, not a supported input class.

### Verification

`tools/server_smoke.sh` starts the daemon, replays the fixtures through the
socket and compares the text with an in-process CLI run:

```sh
MODEL=x-asr-zh-en-q8_0.gguf tools/server_smoke.sh "zh.wav en.wav zh-10.wav" CPU
```

Measured end to end, including the socket and the queue, see
[BENCHMARK.md](BENCHMARK.md#server-end-to-end).

## Inference flow

| Stage | Execution |
| --- | --- |
| PCM to 80-bin Kaldi fbank | CPU, incremental as packets arrive |
| Conv2d subsampling, ConvNeXt, six Zipformer stacks, encoder projection | Selected GGML device |
| Encoder state update | Same device, as copy nodes at the end of the encoder graph |
| Transducer predictor, joiner, greedy token selection and text formatting | CPU |

The encoder consumes 61 feature frames and advances by 48 frames. Each chunk yields 12 encoder frames. The graph is allocated once and reuses the same state tensors. Each chunk transfers its features and masks into the graph and reads the encoder projection back to the CPU. The 115 state tensors stay in the graph between chunks.

The graph uses sum-of-squares and contiguous square root for BiasNorm. It maps the ConvNeXt depthwise convolution to the native depthwise op. Fixed relative-position projections and convolution edge scales are computed once at graph initialization. These changes use the existing weights. Small floating-point differences are possible from reduction order and kernel precision.

For C++ integration, load the GGML backends, open a context, create a stream, and pass normalized 16 kHz mono PCM to `xasr_append`. Call `xasr_commit` at the upstream VAD endpoint and destroy the stream when the transcript is returned. Set `xasr_context_params.n_streams` to open several independent streams in one context: each gets its own encoder graph and state, and all of them share one copy of the weights. `xasr_stream_init` returns null when every slot is busy. The audio limit is ten minutes per stream. The synchronous API runs on the calling thread. The stepping API prepares input once and advances by at most one encoder chunk per call. Serialize operations on a context. The API is in [xasr.h](xasr.h). The CLI exposes the same through `--streams N` and `--concurrent`, which feeds N aligned streams and commits them in order.

The example covers the inference path. A server can put a request queue and IPC around this API. True in-flight batching will require independent persistent stream state and a batched encoder graph.

See [BENCHMARK.md](BENCHMARK.md) for the reproducible device benchmark, fixture hashes, and results. [OPTIMIZATION.md](OPTIMIZATION.md) explains the graph changes, how to isolate each one, and the next experiments. [BUILD.md](BUILD.md) has the host, Android and incremental build steps. [ALIGNMENT.md](ALIGNMENT.md) compares the streaming configuration with sherpa-onnx, and [ONNX-ANALYSIS.md](ONNX-ANALYSIS.md) compares the ONNX graph with this implementation. `tools/` holds the sherpa-onnx control harness, the four-configuration driver and the tail fixture generator.
