# Paraformer (FunASR) example

CPU-first baseline for adapting FunASR Paraformer-large to this llama.cpp fork.
The runtime code is derived from CrispASR's `src/paraformer.cpp` and its
`core/{sanm,kaldi_fbank,lfr,parallel_for}` helpers (FunASR Model License;
commercial use requires attribution). The GGUF loader and device shims here are
minimal local replacements so the example builds against this repository.

## Build

```sh
cmake -S . -B build-x-asr -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build-x-asr --target llama-paraformer -j
```

Android build and push use the standard Snapdragon driver; the binary is
installed as `bin/llama-paraformer`.

## Run

```sh
llama-paraformer MODEL.gguf WAV... [--device HTP0:0] [--threads 2] [--warmup FILE] [--timestamps]
```

Input is 16 kHz mono PCM WAV. Output is TSV compatible with
`examples/x-asr/tools/eval_vehicle.py`:

```text
WAV\taudio_ms=..\twall_ms=..\trtf=..\t[meta=..]\tTRANSCRIPT
```

`--device` selects any backend name accepted by `ggml_backend_init_by_name`;
omit it for pure CPU. `--timestamps` also exposes CIF fire frames through
`paraformer_transcribe_with_timestamps()` (currently only the char count is
printed by the CLI).

### Whole-word / prefix-conditional logit bias (off by default)

`--hotwords FILE` applies a fixed logit gain in the NAR decoder's argmax. Two
line forms (`#` comments OK, gain in [0,5], tokens must exist in the GGUF vocab):

```text
<gain> <token>              global gain at every CIF position
<gain> <prefix>><token>     conditional: only when previous token == prefix
```

The second form handles BPE word-tail swaps, and works only when the right and
wrong words have the **same token count** and differ in the last token —
`se@@ ed`→`se@@ at` ("seed"→"seat") works; `footwall`→`footwell` (3→2 tokens)
and `likes`→`lights` (2→3) cannot be fixed by a bias. Without the flag the
baseline is bit-identical. `hotwords_vehicle.txt` ships the recommended config:

```sh
llama-paraformer MODEL.gguf WAV --hotwords hotwords_vehicle.txt [--device HTP0:0]
```

`--dump-logits N` prints the top-5 logits (with gain applied) at the first N CIF
positions; `PF_WATCH_TOKENS=a,b` (env) appends the exact gap of named tokens.
Use these to measure the gap a bias must overcome before picking a gain — on
this model `said→set` is 2.89–3.00, `too→two` 0.96, `se@@+ed→at` 3.26–3.46,
while `in→and` is 4.16 and also disturbs a neighbouring position, so excluded.

With the existing punctuation layer, the 32-clip cabin command field goes
68.8% → 75.0% (§13, matching x-asr) → **78.1%** (§14, adding `3.8 se@@>at`);
en WER 12.21% → 8.40%, zh CER unchanged, no regressions or false triggers.
See design doc §13–14.

## Status (2026-11-17, host CPU, Q8_0)

- Weight: `cstr/paraformer-zh-GGUF`, Q8_0, sha256 `e73232ec…`.
- 32-clip `tingting` cabin fixture: Chinese CER 0.60%, English WER 12.21%.
- Raw command field 56.2%; after punctuation-only post-processing 68.8%.
  The raw number is depressed because Paraformer emits no punctuation and the
  semantic evaluator splits clauses on `，`/`,`. This is a post-processing gap,
  not an acoustic error.
- x-asr reference on the same fixture (on-arm): command 75.0%, CER 7.23%.
- Device SM6650, 32 clips, warm-run excluded: CPU p50/p95 357/496 ms, HTP
  290/380 ms; transcripts are identical between CPU and HTP.
- **Graph and arena reuse (§16)**: encoder and decoder graphs are now cached per
  exact shape with one scheduler each, and the 256 MB per-call arena zeroing is
  gone. Device HTP: 32-clip p50/p95/p99 280/367/373 ms, 32/32 transcripts
  unchanged; a repeated same-length request drops 306 to 280 ms; peak RSS falls
  from about 700 MB to 252 MB per process, which is what makes a third engine or
  a punctuation model affordable. Cache hit rate on the varied-length field is
  only 3/32, so the average gain is 3-4%.
- **Warmup (M1)**: the HTP first call pays ~1 s in `ggml_backend_sched_alloc_graph`
  (encoder 733 ms + decoder 221 ms; first DSP buffer allocation + graph dispatch).
  `--warmup FILE` runs one full inference off the measured path and cuts the first
  real request from 1130 ms to 185 ms; p99/max over 32 clips 1278→390 ms, p50
  unchanged (291 ms), transcripts 32/32 identical. sched alloc is process-wide
  (a 1.4 s warmup also covers a 3.5 s request), so a single representative clip
  is enough. Length bucketing was tested and rejected: the remaining cost is
  compute linear in LFR frames, which padding cannot reduce.
- Paraformer fixes `Set the driver temperature...` in one clip, but still emits
  `said` in two English seat/light clips; English remains the open risk.

## Server

`llama-paraformer-server` is a whole-utterance ASR daemon. Framing matches the
x-asr client (`<III` request header, `<II` reply header); `RECOGNIZE` (op 0)
carries one float32 16 kHz segment and the reply is `FINAL` (status 1) with the
UTF-8 transcript. The GGUF is read once and shared; each engine materializes its
own weights, so contexts are independent and run in parallel with no global lock.
Warm up every engine before accepting traffic:

```sh
python3 tools/bench_pf_service.py --wavs WAV --emit-warmup warmup.f32
# default: 2 engines (HTP + CPU), shared weights, 2 workers
llama-paraformer-server MODEL.gguf HTP0:0 --port 8876 --warmup warmup.f32 --threads 2 --engines 2
python3 tools/bench_pf_service.py --port 8876 --wavs WAV1,WAV2,WAV3,WAV4 --lanes 1,2,4 --reps 8
```

`--engines 1` gives the old single-context serial behavior (baseline).
SM6650, shared weights load 154 ms, warmup HTP 1386 ms / CPU 872 ms (before accept):

| config | 1 lane p50/p95 | 2 lanes p50/p95 | 4 lanes p50/p95 |
|---|---:|---:|---:|
| 1 engine (HTP, baseline) | 285 / 296 | 439 / 577 | 707 / 1126 |
| 2 engines (HTP+CPU, shared weights) | 337 / 370 | 329 / **373** | 516 / **740** |
| 2 engines + 4 lanes stagger 300 ms | — | — | 310 / **349** |

2-lane p95 drops 577 → 373 ms (−35%) and 4-lane 1126 → 740 ms (−34%); transcripts
are byte-identical across engines (distinct=1 per clip). A single weight buffer
cannot back both backends (ggml binds ops to the weight's physical buffer and HTP
needs tiled repack), so each engine keeps its own buffer — what is shared is the
disk read + GGUF parse. See design doc §12.

### Weight variants

`tools/requant.cpp` rebuilds the published Q8_0 file with ggml's own quantizer
and keeps every non-quantized tensor in its original type:

```sh
build-x-asr/bin/llama-paraformer-requant paraformer-zh-q8_0.gguf out-q4_k.gguf Q4_K
```

Q4_K halves resident memory (252 to 146 MB) and is 17 percent slower on both HTP
and CPU, with transcripts inside quantization noise, so Q8_0 stays the default.
Q5_K/Q6_K use the same tool. See design doc section 24.

## OpenAI-compatible API

`llama-paraformer-api` serves the engine over HTTP with the OpenAI audio shape.
The front end already commits one segment at the VAD endpoint, so a whole
utterance per request maps directly onto the engine.

```sh
build-x-asr/bin/llama-paraformer-api MODEL.gguf --device HTP0:0 --port 8080 \
    --engines 2 --threads 2 --queue 64 --timeout-ms 30000 --model-id paraformer-zh-q8_0 \
    --warmup hvac_en01.wav

curl -sS localhost:8080/health
curl -sS localhost:8080/v1/models
curl -sS -X POST localhost:8080/v1/audio/transcriptions -F file=@a.wav
curl -sS -X POST localhost:8080/v1/audio/transcriptions -F file=@a.wav -F response_format=verbose_json
curl -sS -X POST localhost:8080/asr/segment --data-binary @segment.f32
```

`response_format` accepts `json`, `text`, `verbose_json` and `srt`; anything else is
400. `verbose_json` carries `words` and pause-split `segments` from the CIF token
times, and `srt` emits one block per pause. `punctuate=true` inserts a comma
where the silence between two words crosses `--punct-gap-ms` (600 ms default):
that is acoustic evidence, and it takes the cabin command field from 71.9 to
78.1 percent where the earlier keyword restorer regressed to 46.9. Errors use the OpenAI envelope `{"error": {...}}`. Every reply carries
`X-Queue-Ms` and `X-Infer-Ms` so the tail can be split into queueing and
inference. Overload returns 429 and a queue timeout returns 503. Input must be
16 kHz mono 16-bit PCM WAV; other formats are rejected rather than resampled.
Realtime streaming is the same server on `/v1/realtime` (WebSocket): send
`input_audio_buffer.append` with base64 PCM16 while the user speaks, then
`input_audio_buffer.commit` at the VAD endpoint. The server answers with
`conversation.item.input_audio_transcription.delta` (throttled partials, see
`--partial-ms`/`--partial-interval-ms`) and `.completed`. Partials never delay a
commit: commit tasks jump the engine queue.

```sh
python3 tools/ws_client.py --url ws://127.0.0.1:8080/v1/realtime --wav a.wav --paced
```

See docs/backend/snapdragon/asr-server-api-design.md for the full surface,
including the Realtime event mapping and what a general ASR server needs.

### Speaker gate (optional)

The API can score each committed segment against the enrolled driver's voiceprint
with CAM++ through sherpa-onnx, on the CPU, while the ASR runs on HTP. It is a
compile-time option so a plain build has no sherpa dependency:

```sh
# Android build (sherpa headers and libs mounted into the container)
docker run --rm -v "$PWD:/workspace" -v /tmp/sherpa:/sherpa:ro \
    --platform linux/amd64 -u 501:20 ghcr.io/snapdragon-toolchain/arm64-android:v0.8 \
    bash -c 'cmake -S /workspace -B /workspace/build-android-xasr -DXASR_SPEAKER=ON \
        -DXASR_SHERPA_INCLUDE_DIR=/sherpa/inc -DXASR_SHERPA_LIB_DIR=/sherpa/lib && \
        cmake --build /workspace/build-android-xasr --target llama-paraformer-api -j 6'

# on the phone: LD_LIBRARY_PATH needs lib/ and onnx/bin (sherpa + onnxruntime)
export LD_LIBRARY_PATH=$PWD/lib:$PWD/onnx/bin
bin/llama-paraformer-api MODEL.gguf --device HTP0:0 --port 8899 --engines 2 \
    --speaker-model onnx/model/spk_od.onnx --enroll-dir pf-enroll --speaker-threshold 0.55
```

`--enroll-dir` holds a handful of the driver's 16 kHz WAVs; startup prints the clip
count and the mean self-cosine. Each commit then returns `X-Speaker-Score`,
`X-Speaker-Ok` (REST) or `speaker_score`/`speaker_ok` (verbose_json and the
Realtime completed event). Measured on SM6650: enrolment self-cosine 0.904, driver
0.921, passenger 0.363, and inference stays inside the no-gate noise because the
110 ms embedding overlaps the 320 ms ASR. The score is reported, not enforced.

## Profiling

Capture a per-op HTP profile (one warmup + one measured utterance) and
aggregate it into an encoder/decoder cost table:

```sh
GGML_HEXAGON_PROFILE=1 GGML_LOG_LEVEL=debug \
  llama-paraformer MODEL.gguf WAV --device HTP0:0 --warmup WAV > prof.log 2>&1
python3 tools/profile_ops.py prof.log
```

The M2 measurement on SM6650: encoder is kernel-bound (62.7% MUL_MAT+ADD,
11% FSMN CONV_2D_DW, 6% CONT copies), scaling linearly with LFR frames;
dispatch is only ~7%. See design doc §10 for the full breakdown and why this
rules out cheap runtime-level encoder optimizations on the Q8_0 weights.

See `docs/backend/snapdragon/paraformer-htp-adaptation-design.md` for the
stage-split, length-bucket, overlap/stitch and bilingual plan.
