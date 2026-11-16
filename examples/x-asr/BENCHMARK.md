# X-ASR HTP optimization record

## Setup

The [unified benchmark](#unified-benchmark-onnx-int8-gguf-cpu-gguf-htp) below is
the current reference. The tables in this section and in "Single-stream results"
are the historical record of the first port; they were measured on an earlier
build and are kept for the optimization history.

Measured on SM6650, Android 16, Hexagon v73 (2 HVX units, no HMX, 2 MB VTCM), with the backend from this fork at `6a08e53e7`. The model is [X-ASR-zh-en Q8_0](https://huggingface.co/cstr/x-asr-zh-en-GGUF), SHA-256 `1ca120084a1517cf02d96e44cdd9a9544f0c887f6d0149c9f85d151be6833a61`. Both runs use the same model and greedy decode. GGML CPU uses two encoder threads. Both runs use 480 ms chunks, 20 ms input packets, and 1,200 ms of flush padding supplied immediately at commit.

The corpus is four synthetic cabin commands or command sequences. The long files were cut to 10 or 20 seconds. These are single-stream smoke measurements, with one measured replay per file per mode. The audio excludes real cabin noise and the loop excludes external VAD, queue, and IPC. Treat the table as latency observations on this hardware rather than a production percentile or recognition-rate estimate.

| Fixture | Duration | SHA-256 |
|---|---:|---|
| `zh.wav` | 2.652 s | `e920a925de129a60f7a97beea99cbbdc51058e6736d5f634c9886655c89853fd` |
| `en.wav` | 2.708 s | `375910b5d75a5866a13797f8e4e931acd6ac7172c256b71b869f1773acb9af9e` |
| `zh-10.wav` | 10 s | `97a60a20d37e01209d1fd975efd3ee98b73c0b94aafa79ccc75b69edddddde7a` |
| `en-20.wav` | 20 s | `50de1c6e4e78fd07d4604f00e522b87293369c8fb8dca0ad6a6fd7d4d877e391` |

## Unified benchmark: ONNX INT8, GGUF CPU, GGUF HTP

The four configurations below share one device, one WAV set, one 20 ms packet
feed, one 1,200 ms tail padding, one greedy decoder and one text normalisation.
The ONNX control exists because it is the reference implementation of the same
model geometry; it is not part of the deliverable.

### Configurations

| Config | Weights | Quantisation | Runtime | Threads | Chunk and padding | Backend parameters |
|---|---|---|---|---|---|---|
| `onnx-cpu` | X-ASR 480 ms release ONNX | encoder and joiner operator-level int8, decoder fp32 | sherpa-onnx v1.13.8 C API, onnxruntime 1.28.2 | 2 intra-op | 61 frame window, 48 frame advance, 1,200 ms pad | provider cpu, greedy_search, endpoint off |
| `gguf-cpu` | `x-asr-zh-en-q8_0.gguf` | Q8_0 | this fork, `llama-x-asr` | 2 encoder threads | same | device `CPU` |
| `gguf-htp-wait` | same GGUF | Q8_0 | same | 2 host threads | same | `HTP0:0`, OPBATCH 2048, OPPOLL 0, power burst |
| `gguf-htp-poll` | same GGUF | Q8_0 | same | 2 host threads | same | `HTP0:0`, OPBATCH 2048, OPPOLL 1, power burst |

Both GGUF rows load one weight file. The ONNX row is a different export and a
different quantisation path, so its transcript is not expected to match the GGUF
one character for character; see [Quality](#quality-and-known-differences).

Input hashes: GGUF `1ca120084a1517cf02d96e44cdd9a9544f0c887f6d0149c9f85d151be6833a61`,
`encoder.int8.onnx` `908596dcc137a73b95be908ca55e88caa1b3dbbe8027c171615f4b0609c5eb1e`,
`decoder.onnx` `a1cbc9eac2d5e3fb6617a218c67ad6daaa7f4e0fd225f08b2c22ab0413c8c257`,
`joiner.int8.onnx` `aedb7fa697b2ab43f20499826fff7c997eea7d67db77be97769aeeeb726e63b3`.
WAV hashes are in the [Setup](#setup) table.

### Method

```sh
# GGUF: one process per configuration per round, all fixtures in that process.
adb shell -T "cd /data/local/tmp/llama.cpp && \
  export LD_LIBRARY_PATH=\$PWD/lib ADSP_LIBRARY_PATH=\$PWD/lib \
  GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_OPPOLL=<0|1> \
  GGML_HEXAGON_POWER_MODE=burst XASR_TIMING=1 && \
  bin/llama-x-asr x-asr-zh-en-q8_0.gguf <CPU|HTP0:0> zh.wav en.wav zh-10.wav en-20.wav --realtime"

# ONNX control: same 20 ms feed and the same 1,200 ms padding.
adb shell -T "LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/onnx/bin \
  /data/local/tmp/llama.cpp/onnx/bin/onnx-bench /data/local/tmp/llama.cpp/onnx/model 2 1 \
  zh.wav en.wav zh-10.wav en-20.wav"
```

`examples/x-asr/tools/bench_ab.py` runs all four configurations and writes the
tables below. Rounds reverse the configuration order to spread device drift, and
each round starts a fresh process per configuration.

```sh
python3 examples/x-asr/tools/bench_ab.py \
    --adb "$ANDROID_SDK/platform-tools/adb" --serial "$SERIAL" \
    --remote /data/local/tmp/llama.cpp \
    --onnx-dir /data/local/tmp/llama.cpp/onnx/model --onnx-threads 2 \
    --wavs /data/local/tmp/llama.cpp/{zh,en,zh-10,en-20}.wav \
    --rounds 3 --log-dir /tmp/xasr-ab-1
```

`tools/onnx-bench.cpp` is the ONNX control. It uses the same 20 ms pacing loop as
`main.cpp` and deliberately appends the same 1,200 ms of zeros, because
sherpa-onnx itself never pads at end of utterance and loses the last 1 to 48
frames, including whole trailing words on three of four fixtures
([ALIGNMENT.md](ALIGNMENT.md), D1). The measurement therefore compares runtimes,
not the two projects' default end-of-utterance policies.

Raw logs for the run quoted here:
`$HOME/Library/Caches/qwen3-asr-htp/xasr-ab-2026-11-17/` (`raw.jsonl`,
`summary.md`, one stderr log per configuration and round).

### Endpoint to final

Snapshot at commit `e1ef64e79`. Later optimization batches report their own A/B
tables and are not folded into this one, so that every row comes from a single
session on the same build.

Tail is measured from the scheduled arrival of the last 20 ms packet to the final
text. Three rounds per fixture, so N=3 per cell.

| Config | zh 2.65 s | en 2.71 s | zh 10 s | en 20 s | chunk p50 | chunks | commit chunks |
|---|---:|---:|---:|---:|---:|---:|---:|
| `onnx-cpu` | 126.0 | 121.8 | 169.1 | 125.0 | 62.3 | 7 / 7 / 23 / 43 | 2 / 2 / 3 / 2 |
| `gguf-cpu` | 161.4 | 169.7 | 252.6 | 181.6 | 75.0 | same | same |
| `gguf-htp-wait` | 296.1 | 259.0 | 380.9 | 270.0 | 122.5 | same | same |
| `gguf-htp-poll` | 189.4 | 192.3 | 282.3 | 196.4 | 109.2 | same | same |

The joiner projection and its argmax now run on the selected backend for every
row above, which is [H5](OPTIMIZATION.md#h5-joiner-projection-and-argmax-on-the-selected-backend-kept-now-the-default).
Compared with the same matrix before that change, HTP polling moved from 200.0 /
200.4 / 303.3 / 200.2 to the values above; the paired A/B is in H5 because
device drift between the two sessions is of the same order as the change.

P95 over the same three samples: ONNX 126.7 / 122.5 / 171.8 / 129.6, GGUF CPU
164.6 / 170.8 / 265.7 / 183.8, HTP polling 191.5 / 193.0 / 282.5 / 196.6. With
N=3 these are max values, not steady-state percentiles.

The chunk count is identical in all four configurations on all four fixtures:
the encoder geometry, the readiness rule and the padding policy match. The
"commit chunks" column is the number of encoder chunks that run inside
`xasr_commit`, and it is the part of the tail that a faster chunk or a shorter
tail policy can remove.

### Normalised per-chunk cost

Summed over all runs of a configuration, so long clips carry their own weight.

| Config | chunk ms | greedy ms/frame | fbank ms per audio s | submit ms/batch | wait ms/batch | host CPU ms per audio s | append work / audio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `onnx-cpu` | 62.5 | n/a | n/a | n/a | n/a | n/a | n/a |
| `gguf-cpu` | 76.4 | 1.048 | 12.1 | n/a | n/a | 404 | 17.8% |
| `gguf-htp-wait` | 122.3 | 0.616 | 20.5 | 0.17 | 50.28 | 76 | 31.1% |
| `gguf-htp-poll` | 97.2 | 0.339 | 11.3 | 0.15 | 42.94 | 253 | 22.0% |

The "greedy" column now contains the backend joiner plus the host predictor
update, so it is the whole decoder cost, not just the host loop. Summed over the
whole matrix, one joiner call costs 1.07 ms on `gguf-htp-poll`, 1.74 ms on
`gguf-htp-wait` (the blocking completion dominates) and 3.56 ms on `gguf-cpu`,
for 810 calls and 2,880 encoder frames per configuration. The host loop it
replaced measured 0.64 to 0.70 ms per frame on this device, and one call covers
about five frames, so the HTP call is roughly three times cheaper and the CPU
backend call is a wash.

"submit" and "wait" are the host-side cost of handing one DSP command batch to
the DSP and of collecting its completion, accumulated in the backend and read
per chunk. A chunk is two batches at OPBATCH 2048. "append work / audio" is the
sum of the time spent inside `xasr_append` divided by the audio duration; below
100% the pipeline keeps up with real time and the tail is only the commit work.

ONNX row: the sherpa-onnx C API does not expose its internal submit or wait, and
its feature extraction runs inside the same call, so those columns stay empty
rather than being estimated.

### Where the tail goes

Every configuration runs 7 chunks on the 2.65 s clip and 2 of them after commit,
so the tail is two encoder chunks plus the decoder for their 24 encoder frames.
The commit wall time is within a few milliseconds of the tail on every row below,
which means the pipeline is not backlogged when the last packet arrives and
there is no queue to drain.

| Config | commit p50, zh | commit p50, zh 10 s | commit chunks |
|---|---:|---:|---:|
| `onnx-cpu` | 124.2 | 167.3 | 2 / 3 |
| `gguf-cpu` | 160.8 | 251.2 | 2 / 3 |
| `gguf-htp-poll` | 180.1 | 278.3 | 2 / 3 |
| `gguf-htp-wait` | 256.3 | 380.2 | 2 / 3 |

The per-encoder-chunk DSP cost is measured separately in the operator profile
(85.7 ms after H2), and the decoder contributes 2 to 3 joiner calls per chunk:
18.4 ms for 16 calls on the 2.65 s clip in H5. Commit bookkeeping is not visible
at this scale: feature and mask upload are 0.3 ms per chunk, the encoder
projection read-back is 0.06 ms, detokenisation is 0.05 ms.

### HTP op fusion A/B

One factor changed between the two arms: the HVX matmul-plus-bias fusion. Same
build flags, same device, same fixtures, same backend parameters.

| Arm | DSP ops per chunk | batch 1 | batch 2 | DSP ms/chunk | zh tail | zh 10 s tail | text |
|---|---|---:|---:|---:|---:|---:|---|
| before | 2,820 (2048+772) | 67.6 ms | 21.0 ms | 88.6 | 190.9, 191.1 | 295.9, 297.2 | reference |
| after | 2,531 (2048+483) | 71.7 ms | 13.8 ms | 85.7 | 188.4, 188.7 | 293.8, 294.5 | identical |

Each tail cell is one paced run; the two values are two repetitions. DSP batch
times are the median of 14 batches from `GGML_HEXAGON_PROFILE=1` replays of two
clips, and the profile run itself is not used for the latency columns. See
[OPTIMIZATION.md](OPTIMIZATION.md#h2-fuse-the-bias-add-into-the-hvx-matmul-kept)
for the change and its limits.

A separate experiment replaced the completion wait with a bounded timeout to cut
host CPU. It was rejected: every finite timeout behaved like blocking. The data
is in [OPTIMIZATION.md](OPTIMIZATION.md#h1-bounded-wait-instead-of-spin-for-dsp-completion-rejected-no-code-kept).

### HTP ConvNeXt quantisation A/B

One factor changed: the two ConvNeXt pointwise projections, F32 in the GGUF,
replaced by Q8_0 copies of the same weights.

| Arm | weights | DSP ms/chunk | projections ms/chunk | zh | en | zh 10 s | en 20 s |
|---|---|---:|---:|---:|---:|---:|---:|
| F32 | 196 KB | 86.9 | 6.09 | 179.9 | 193.4 | 282.6 | 195.1 |
| Q8_0 | 104 KB | 81.3 | 1.11 | 170.7 | 184.2 | 265.5 | 183.6 |

One paced run per cell, identical text in both arms and on both backends. The
DSP operator profile confirms the placement: `MUL_MAT+ADD f32 x f32 x f32`
disappears and `MUL_MAT+ADD q8_0` gains 28 ops per 14 chunks. See
[OPTIMIZATION.md](OPTIMIZATION.md#h6-quantise-the-two-convext-pointwise-projections-kept-now-the-default).
The joiner A/B is in
[H5](OPTIMIZATION.md#h5-joiner-projection-and-argmax-on-the-selected-backend-kept-now-the-default).

### Backend placement and DSP batches

All four GGUF runs report `encoder HTP0:0 2877/2877 ops, 1 scheduler splits,
4148736 bytes of cache` and `joiner HTP0:0 5/5 ops, weight q8_0 512 x 5000, 12
frames per call`, so the whole encoder and the whole joiner graph are placed on
the DSP with no fallback. One encoder scheduler split still produces two DSP
command batches per chunk (2,048 + 483 ops after H2) at OPBATCH 2048. A joiner
call is a single additional batch. The completion wait is the
largest single host cost: 43 ms per batch while polling and 50 ms per batch while
blocked, against roughly 44 ms of DSP work per batch. That gap is the subject of
the [poll wait experiment](OPTIMIZATION.md#h1-bounded-wait-instead-of-spin-for-dsp-completion-rejected-no-code-kept).

### Quality and known differences

| Fixture | `onnx-cpu` | GGUF |
|---|---|---|
| zh.wav | 擎导航到上海虹桥机场 | 请导航到上海虹桥机场 |
| zh-10.wav | 擎导航到上海虹桥机场，途经最近的充电站，到了之后提醒我给陈佳宁打电话，车内温度调到二十二 | 请导航到上海虹桥机场，途经最近的充电站，到了之后提醒我给陈佳宁打电话，车内温度调到二十二 |
| en.wav | said the cabin temperature to twenty two degrees | same |
| en-20.wav | ... remind me to call Sarah. Set the cabin temperature ... | ... remind me to call Sarah, said the cabin temperature ... |

The ONNX int8 export drops the leading syllable 请 on both Mandarin fixtures and
changes "Sarah. Set" into "Sarah, said" on the 20 s English fixture. The two GGUF
backends agree with each other on every fixture and every round. This is the
opposite of what a latency-only reading of the table would suggest, and it is the
reason the ONNX row is a latency reference and not a candidate deployment.
[ALIGNMENT.md](ALIGNMENT.md) D6 quantifies the same effect from the sherpa-onnx
side: character error 1.9%, 0%, 6.7%, 7.9% on the four release fixtures when only
the weight precision differs.

### Startup and per-stream cost

`xasr: init` is model load plus graph allocation, `xasr: stream init` is one
stream creation. Measured on the current build with `XASR_TIMING=1`, one process
per row, two streams created back to back in that process.

| Stage | HTP polling | CPU |
|---|---:|---:|
| model init (cold) | 777.6 ms | 228.3 ms |
| first stream | 1.00 ms | 1.92 ms |
| second stream | 0.61 ms | 0.52 ms |

Stream creation clears 4,148,736 bytes of encoder state, so the per-request
reset is under 2 ms and per-stream state is 4 MB. For a server the constraint is
therefore memory and DSP residency, not the reset cost. HTP pays about 550 ms
more on the first request because the model, the tiled weight repack and the
graph allocation all land on the DSP.

### Limitations

- N=3 per cell. The P95 column is a maximum, not a percentile. Phase 3 owns the
  repeated-measurement percentiles and the concurrent-load cases.
- One stream at a time, no external VAD, no queue and no IPC. The tail excludes
  ingress latency before the first packet and all service overhead.
- Device state is not fully controlled. The historical table below records
  275 ms for the same CPU configuration; the same binary today measures 242 ms
  because the device was warmer during the earlier sweep. Only same-build,
  interleaved comparisons in this section are used to rank configurations.
- The build configuration changed between the historical table and this one. The
  current Android build uses the Snapdragon release preset, which sets
  `-march=armv8.7a+fp16+dotprod+i8mm -flto -ffp-model=fast`. The historical
  build had empty `CMAKE_CXX_FLAGS`. That accounts for most of the CPU
  difference between the two tables, and it is why this section reports numbers
  from one build only.
- `ru_maxrss` is KiB on Android and bytes on macOS; do not mix hosts.
- The DSP operator profile perturbs timing. All latency numbers here come from
  runs without `GGML_HEXAGON_PROFILE`.

## Concurrent streams

The context can hold one encoder graph and state set per stream, all sharing one
copy of the weights. `--streams N --concurrent` feeds the same packets to N
streams, aligns their endpoints, and commits them in slot order, so stream i
pays the commits of the streams before it. That is the head-of-line cost of a
single worker and the baseline for scheduling work.

HTP polling, OPBATCH 2048, the 2.65 s Mandarin clip, one paced run per row:

| Streams | endpoint to final, in commit order (ms) | max | model init | peak RSS |
|---:|---|---:|---:|---:|
| 1 | 169.6 | 169.6 | 819 ms | 61.0 MB |
| 2 | 169.4, 340.5 | 340.5 | 831 ms | 78.0 MB |
| 4 | 170.0, 341.0, 512.2, 683.0 | 683.0 | 881 ms | 112.2 MB |

Each extra stream adds about 171 ms, which is one full commit: two encoder
chunks plus the decoder. Every stream returns the same text as the single-stream
run, and the same numbers reproduce when the four slots are reused by a second
and third wave in one process, including a language change, so the per-stream
state is cleared on reuse. Cost per extra slot is about 17 MB of memory and 20 ms
of model init.

What this does not do: the streams are serialized. Total work for four aligned
endpoints is four commits, so the last stream waits for the other three. Cutting
that needs the encoder batched across streams, which needs a graph with a stream
dimension through the attention and its per-stream left context. Until then the
four-way simultaneous endpoint number is 683 ms, not 170 ms.

## Reproduce on Android

Run these blocks from the repository root in the same Bash session. The [Snapdragon toolchain guide](../../docs/backend/snapdragon/README.md) covers Docker and ADB setup. Set `ANDROID_SERIAL` if several devices are attached.

### 1. Deploy the example and inputs

```bash
set -euo pipefail
ADB=${ADB:-adb}
REMOTE=${REMOTE:-/data/local/tmp/llama.cpp}
LOG_DIR=${LOG_DIR:-/tmp/xasr-bench-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$LOG_DIR"

# build.py invokes adb by name for deployment.
export PATH="$(dirname "$(command -v "$ADB")"):$PATH"
python3 scripts/snapdragon/build.py --target "android${ANDROID_SERIAL:+:$ANDROID_SERIAL}" --target-dir "$REMOTE" --push
"$ADB" shell -T 'getprop ro.soc.model; getprop ro.build.version.release' > "$LOG_DIR/device.txt"
git rev-parse HEAD > "$LOG_DIR/backend-revision.txt"
shasum -a 256 examples/x-asr/*.cpp examples/x-asr/*.h examples/x-asr/CMakeLists.txt > "$LOG_DIR/source-sha256.txt"
```

The original files are available on the development host. Use the exact files below for historical comparison. The model can also be downloaded using the [README command](README.md#build-and-run-on-cpu). The audio stays outside the source tree.

```bash
CACHE="$HOME/Library/Caches/qwen3-asr-htp"
MODEL="$CACHE/xasr-multistream/x-asr-zh-en-q8_0.gguf"
AUDIO_DIR="$CACHE"
LONG_AUDIO_DIR="$CACHE/cabin-asr-probe/long-perf"

shasum -a 256 "$MODEL" "$AUDIO_DIR/zh.wav" "$AUDIO_DIR/en.wav" \
    "$LONG_AUDIO_DIR/zh-10.wav" "$LONG_AUDIO_DIR/en-20.wav" | tee "$LOG_DIR/input-sha256.txt"
"$ADB" push "$MODEL" "$AUDIO_DIR/zh.wav" "$AUDIO_DIR/en.wav" \
    "$LONG_AUDIO_DIR/zh-10.wav" "$LONG_AUDIO_DIR/en-20.wav" "$REMOTE/"
"$ADB" shell -T "cd '$REMOTE' && sha256sum x-asr-zh-en-q8_0.gguf zh.wav en.wav zh-10.wav en-20.wav" \
    > "$LOG_DIR/device-input-sha256.txt"
```

If the original clips are unavailable, prepare two short commands in Mandarin and English, one 10 s Mandarin clip, and one 20 s English clip. Convert each input with `ffmpeg -i input.wav -ar 16000 -ac 1 -c:a pcm_s16le output.wav`, using the four names listed above. Record the new WAV hashes and use the same files in every configuration. This reproduces the measurement procedure. The historical transcript and latency rows apply to the original hashes.

### 2. Collect tail, RTF and polling data

```bash
run_xasr() {
    local label=$1 device=$2 batch=$3 poll=$4
    shift 4
    "$ADB" shell -T "cd '$REMOTE' &&
        export LD_LIBRARY_PATH=\$PWD/lib ADSP_LIBRARY_PATH=\$PWD/lib &&
        export GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=$batch GGML_HEXAGON_OPPOLL=$poll &&
        export GGML_HEXAGON_POWER_MODE=burst GGML_HEXAGON_PROFILE=0 &&
        time -p bin/llama-x-asr x-asr-zh-en-q8_0.gguf $device $*" \
        > "$LOG_DIR/$label.tsv" 2> "$LOG_DIR/$label.log"
}

clips=(zh.wav zh.wav en.wav zh-10.wav en-20.wav)
run_xasr cpu-rt      CPU    2048 0 "${clips[@]}" --realtime
run_xasr htp-wait-rt HTP0:0 2048 0 "${clips[@]}" --realtime
run_xasr htp-poll-rt HTP0:0 2048 1 "${clips[@]}" --realtime
run_xasr cpu-rtf      CPU    2048 0 "${clips[@]}"
run_xasr htp-wait-rtf HTP0:0 2048 0 "${clips[@]}"
run_xasr htp-poll-rtf HTP0:0 2048 1 "${clips[@]}"

for batch in 2048 4096; do
    for poll in 0 1; do
        run_xasr "sweep-$batch-$poll" HTP0:0 "$batch" "$poll" zh.wav zh.wav en.wav zh.wav en.wav
    done
done
```

Run each case in a fresh process and discard the first clip for latency and RTF. Keep the same power mode, device state, and background load across runs. For comparison across builds, repeat the matrix and alternate CPU and HTP run order. Test the scheduler with the placement log even when only latency is being compared.

### 3. Check placement and summarize

The current graph should log `encoder HTP0:0 2877/2877 ops, 1 scheduler splits, 4148736 bytes of cache`. If a graph change alters the node count, require equal assigned and total counts and inspect the number of splits. The standard example returns an error on encoder fallback. A separate per-op profile can perturb latency, so collect it in a separate pass.

```bash
python3 - "$LOG_DIR" <<'PYTHON'
import re
import sys
from pathlib import Path
from statistics import median

root = Path(sys.argv[1])
clips = ["zh.wav", "zh.wav", "en.wav", "zh-10.wav", "en-20.wav"]
reference = None
for prefix in ("cpu", "htp-wait", "htp-poll"):
    for mode, key in (("rt", "tail_ms"), ("rtf", "rtf")):
        name = f"{prefix}-{mode}"
        rows = [line.split("\t") for line in (root / f"{name}.tsv").read_text().splitlines()]
        assert [row[0] for row in rows] == clips, name
        assert rows[0][-1] == rows[1][-1], f"repeated clip changed: {name}"
        text = [(row[0], row[-1]) for row in rows[1:]]
        if reference is None:
            reference = text
        assert text == reference, f"transcript mismatch: {name}"
        log = (root / f"{name}.log").read_text()
        assigned, total = re.search(r"encoder .+ (\d+)/(\d+) ops", log).groups()
        assert assigned == total and int(total) > 0, f"encoder fallback: {name}"
        values = [dict(field.split("=", 1) for field in row[1:-1])[key] for row in rows[1:]]
        print(name, key, ", ".join(f"{row[0]}={value}" for row, value in zip(rows[1:], values)))

for batch, poll in ((2048, 0), (2048, 1), (4096, 0), (4096, 1)):
    source = root / f"sweep-{batch}-{poll}.tsv"
    rows = [line.split("\t") for line in source.read_text().splitlines()]
    assert [row[0] for row in rows] == ["zh.wav", "zh.wav", "en.wav", "zh.wav", "en.wav"], source
    wall = [float(dict(item.split("=", 1) for item in row[1:-1])["wall_ms"]) for row in rows[1:]]
    print(source.stem, "median_wall_ms", round(median(wall), 1))
PYTHON
```

`time -p` is recorded in each `.log`. Add its `user` and `sys` fields for process CPU time across the whole run, including initialization and warmup. For actual service measurements, also report ingress-to-commit delay, commit-to-final delay, and endpoint-to-final delay separately.

## Historical artifacts

On the original development host, `$HOME/Library/Caches/qwen3-asr-htp/xasr-multistream` contains:

| Stage | Files |
|---|---|
| Initial mixed encoder | `xasr-htp-probe.cpp`, `htp-hybrid.log`, `htp-hybrid.tsv`, `cpu-hybrid.tsv` |
| Per-chunk timing before graph changes | `xasr-htp-timing.cpp`, `cpu-timing.log`, `htp-timing.log` |
| BiasNorm-only rewrite | `xasr-htp-rewrite.cpp`, `rewrite.log`, `rewrite.tsv` |
| Persistent graph before constant folding | `example-fixed-CPU.*`, `example-fixed-HTP0:0.*` |
| Current graph and polling | `final-*.tsv`, `offline-*.tsv`, `check-*.tsv`, `tune-*.tsv`, and matching `.log` files |

The old probes are scratch artifacts. The commands above rebuild and benchmark the current example. The [optimization notes](OPTIMIZATION.md) describe which changes to revert for a controlled ablation.

## Graph changes

| Version | HTP scheduler splits | Encoder compute nodes on CPU | Change |
|---|---:|---:|---|
| Initial GGUF probe | 43 | 41 | Host round trip for all state every chunk. BiasNorm and one ConvNeXt convolution matmul fall back. |
| BiasNorm probe | 3 | 1 | Sum rows, take square root on a contiguous vector, and fold the channel count into the scalar multiplier. |
| Persistent graph | 1 | 0 | Use the native depthwise convolution and copy updated state into the device input tensors at the end of the graph. |
| Example with constant folding | 1 | 0 | Precompute positional projections and convolution edge scales. Total scheduled compute nodes drop from 3,105 to 2,877. |

The example places all 2,877 encoder compute nodes on HTP, including cache updates and the encoder projection. It retains 4,148,736 bytes of state between chunks. These node counts include graph copy nodes and exclude view-only nodes. The HTP submitter can divide the scheduler split into several DSP command batches.

## Single-stream results

Tail runs are timed from the scheduled arrival of the last 20 ms packet to final. RTF is processing wall time divided by audio duration in the unpaced run. Model initialization is excluded.

| Audio | Prior HTP probe tail | Example CPU tail | Example HTP default tail | CPU RTF | HTP default RTF |
|---|---:|---:|---:|---:|---:|
| Chinese 2.65 s | 684.1 ms | 275.1 ms | 404.4 ms | 0.321 | 0.398 |
| English 2.71 s | 623.9 ms | 251.8 ms | 345.3 ms | 0.316 | 0.376 |
| Chinese 10 s | 736.9 ms | 386.7 ms | 474.2 ms | 0.279 | 0.334 |
| English 20 s | 625.7 ms | 259.5 ms | 337.4 ms | 0.262 | 0.315 |

For all four clips the text agrees between CPU and HTP, between unpaced and realtime replay, and with the original GGUF probe. The repeated Mandarin short command produces the same text on both runs. The observed short outputs were `请导航到上海虹桥机场` and `said the cabin temperature to twenty two degrees`. These are parity references. The English request begins with “set”, so parity alone does not establish recognition accuracy.

## DSP completion polling

An unpaced short-clip sweep discarded the first clip for each setting. The remaining four clips gave these median processing times.

| HTP command batch limit | Completion polling | Median wall time |
|---:|---:|---:|
| 2048 | off | 1011 ms |
| 4096 | off | 1030 ms |
| 2048 | on | 815 ms |
| 4096 | on | 842 ms |

With the 2048 limit and polling enabled, a second realtime sweep produced:

| Audio | CPU tail (rerun) | HTP polling tail |
|---|---:|---:|
| Chinese 2.65 s | 283.9 ms | 261.0 ms |
| English 2.71 s | 250.6 ms | 241.6 ms |
| Chinese 10 s | 379.1 ms | 357.3 ms |
| English 20 s | 251.7 ms | 244.3 ms |

On the five-playback sequence including warmup, Android `time -p` reported 19.98 CPU seconds on the CPU path and 12.20 CPU seconds on the polling HTP path (user plus system). Polling trades queue wait time for host polling. Recheck this setting with concurrent ASR streams, background CPU load, and thermal throttling before treating the single-stream advantage as a service-level result. The example leaves backend settings under the caller's control.

## Verification

On 2026-11-17 the measurement, transcript comparison, and profile commands were replayed against the existing Android CMake build on the same SM6650. Device input hashes matched the original fixtures. All six latency/RTF runs and all four polling-sweep cases completed. The four real-time HTP polling tails were 265.8, 233.2, 354.6, and 245.7 ms in fixture order. Raw data on the development host is under `$HOME/Library/Caches/qwen3-asr-htp/xasr-multistream/doc-replay-2026-11-17.5xbW7y`.

- Native macOS CPU build and repeated Chinese/English replay.
- Android ARM64 CPU and HTP replay with the same GGUF and matching transcripts in both replay modes.
- Incremental fbank is bit-identical to one-shot fbank on a 20-second synthetic signal with packet sizes of 1, 160, 320, 959 and 16,000 samples.

## Optional HTP operator profile

In the same Bash session, run a separate short replay with HTP profiling enabled. The timing table uses the unprofiled runs above.

```bash
"$ADB" shell -T "cd '$REMOTE' &&
    export LD_LIBRARY_PATH=\$PWD/lib ADSP_LIBRARY_PATH=\$PWD/lib &&
    export GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_POWER_MODE=burst &&
    export GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_PROFILE=1 &&
    bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 zh.wav zh.wav" \
    > "$LOG_DIR/profile.tsv" 2> "$LOG_DIR/profile.log"
python3 scripts/snapdragon/ggml-hexagon-profile.py "$LOG_DIR/profile.log" --top 15 | tee "$LOG_DIR/profile-summary.txt"
```

Use the aggregate DSP time to rank operators and the host-side timing to identify submit, wait, or readback overhead. The operator profile includes both clips. Keep the raw log so a later comparison can account for warmup and repeated executions.

In that short-clip profile, each chunk produced two DSP command batches, with 2,048 and 772 commands. Their reported averages were 66.53 and 20.61 ms per batch. Within the per-operator rows, the 7×7 ConvNeXt depthwise kernel averaged 5.58 ms and its two pointwise matrix products averaged 3.58 and 2.27 ms. The batch rows already include operator time. The profile makes this block a concrete kernel candidate while the unprofiled end-to-final benchmark remains the measure of impact.
