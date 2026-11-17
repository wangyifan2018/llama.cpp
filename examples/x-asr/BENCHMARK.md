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

Staggered endpoints, same device and settings. `--hold-ms T` stops feeding every
stream except the last at `T` before the end, then commits the finished ones
while the last one keeps recording. The short streams therefore transcribe a
truncated utterance; only their latency is comparable.

| Case | endpoint to final (ms) | last stream's own single-stream cost |
|---|---|---|
| 4 aligned, zh 2.65 s | 169.6, 339.5, 510.1, 680.7 | 170 |
| 3 end at 8.5 s, 1 records to 10 s | 171.4, 343.2, 514.9, **269.6** | 265 |
| 2 streams, 3 s stagger, 10 s clip | 171.9, **268.9** | 265 |

When the endpoints do not collide, the continuing stream pays nothing for the
other commits: the DSP drains them while it is still receiving audio, and its
endpoint to final is its own single-stream cost (269.6 against 265, 268.9
against 265). The head-of-line penalty appears only when endpoints are aligned.
That is the case the four-way number above measures, and the reason the next step
is encoder batching rather than scheduling.

What batching needs: a stream dimension through the attention and its per-stream
left context. The feed-forward, convolution and pointwise blocks can be widened
across streams without that, so a partial batch is possible first.

## Server: end to end

These numbers predate encoder batching and the cost-ranked placement rule. For
the current default deployment and its four-way tail, see the
[stage baseline](OPTIMIZATION.md#stage-baseline-after-six-batches); the rows
below remain the reference for the pre-batching server.

`llama-x-asr-server` on the device (HTP polling, 4 slots, 64 request queue),
client on the host through `adb forward`, so every number includes the socket,
the queue and the USB path. Twelve paced repetitions per scenario, sampled with
`tools/bench_service.py`; the fixtures are the ones in the Setup table.

| Scenario | arrivals | N | endpoint to final p50 | p95 | p99 | max |
|---|---|---:|---:|---:|---:|---:|
| single | one 2.65 s clip | 12 | 167.3 | 175.6 | 177.1 | 177.4 |
| aligned4 | four 2.65 s clips, same endpoint | 48 | 638.3 | 941.4 | 942.4 | 943.1 |
| stagger4 | three 8.5 s plus one 10 s, staggered | 48 | 299.2 | 676.3 | 682.0 | 683.4 |
| mixed4 | two 2.65 s, one 10 s, one 20 s | 48 | 254.8 | 564.1 | 596.9 | 598.4 |

The commits in the aligned four-way case, split into time waiting in the queue
and time running the engine, from the server's own log (48 commits):

| Stage | p50 | p95 | max |
|---|---:|---:|---:|
| queue wait | 470.8 | 775.2 | 776.9 |
| commit (engine) | 170.8 | 171.4 | 171.7 |

This is the attribution that matters for the service: with four aligned
endpoints the engine time is flat at 171 ms, the same as a single stream, and
everything above it is queueing. Across all four scenarios (156 commits) the
queue wait is p50 168.2 and p95 768.1, against commit p50 171.0 and p95 267.1.

Single-stream service cost is therefore negligible: 167.3 ms through the socket
against 169.6 ms in-process. The tail in the concurrent scenarios is capacity,
not latency, and the next lever is throughput of the shared encoder, not a
faster commit.

### Hybrid engines

The queue wait above is time the single worker spends inside another stream's
commit, while the device CPU sits idle. `--cpu-streams N` adds a second engine
on the CPU backend with its own worker and queue, and sessions are pinned at
OPEN to the least loaded engine that still has a slot.

Eight paced repetitions per row, aligned four-way (N=32) and single (N=8):

| Engines | aligned4 p50 | p95 | single p50 | peak RSS | commits HTP / CPU |
|---|---:|---:|---:|---:|---:|
| HTP x4 | 636.5 | 944.0 | 166.5 | 112 MB | 40 / 0 |
| HTP x2 + CPU x2 | 302.2 | 433.0 | 165.0 | 313 MB | 24 / 16 |

Halving the worst case comes from using a second processor, not from faster
inference: the engine time per commit is unchanged. The cost is a second copy of
the weights for the CPU engine, about 200 MB of host memory, and the CPU engine
competes with feature extraction and the predictor in the same process.

The commit counts in that table are 24 device and 16 CPU, so the split was 3:2
rather than the 2+2 the paragraph above describes: the router compared queue
depth, and a worker drains its queue before the next OPEN is routed. Placement
now ranks by what a lane costs instead, so an odd device lane is paired before a
third lane starts a step and a second CPU lane is parallel. With that rule and
encoder batching in the graph, four aligned segments of 2.2 to 2.7 s go from
1,614 ms p95 on the device alone to 1,022 ms on this deployment, and the explicit
2+2 configuration measures 995 ms
([OPTIMIZATION.md](OPTIMIZATION.md#h9-batching-stops-paying-at-a-pair-and-the-tail-needs-the-second-engine)).
The rows above predate both changes and are kept as the measurement that
motivated them; one device slot per stream is now the slower configuration.

Peak RSS is 61 MB with one slot and 112 MB with four. Throughput was not
measured under free-running load: every scenario above is paced at real time,
so the numbers show latency under a given arrival pattern, not capacity. Both
runs' raw data: `/tmp/svc-1.jsonl`, `/tmp/svc-aligned.jsonl`, and the server
log on the device, `/data/local/tmp/llama.cpp/server.log`.

Reproduce:

```sh
python3 examples/x-asr/tools/bench_service.py --port 18765 --reps 12 \
    --short zh.wav --long zh-10.wav --long2 en-20.wav \
    --out /tmp/svc-1.jsonl
```

Run-to-run drift is visible in the aligned case: two runs of the same scenario
gave p50 592.4 and 638.3, which is why the plan calls for a fixed arrival
sequence and many more repetitions before any scheduling change is judged.



## Capacity and saturation

Throughput is measured unpaced: the client sends each clip as fast as the socket
accepts, so the number is engine capacity, not a latency test. One process per
cell, 24 clips of 2.65 s per cell, audio seconds per wall second.

| Engines | 1 session | 2 sessions | 4 sessions |
|---|---:|---:|---:|
| HTP x4 | 4.23 | 4.27 | 4.28 |
| HTP x2 + CPU x2 | 4.23 | 6.65 | 8.27 |

One HTP engine saturates at 4.3 times real time no matter how many sessions feed
it, because the DSP is the only worker. Adding the CPU engine lifts the same
device to 8.3 times real time at four sessions, which is the capacity the hybrid
split is buying. A single session still lands on the device engine and is
unchanged.

Saturation latency, four sessions unpaced, five consecutive blocks of 72 clips
(about 4.5 minutes of sustained load):

| Block | endpoint to final p50 | p95 | max |
|---|---:|---:|---:|
| 1 | 709 | 1243 | 1248 |
| 2 | 1028 | 1247 | 1307 |
| 3 | 1036 | 1247 | 1252 |
| 4 | 1035 | 1252 | 1282 |
| 5 | 1026 | 1247 | 1255 |

Driven at capacity the service settles at about 1.0 s p50 and 1.25 s p95 and does
not drift across the blocks, so there is no thermal degradation visible in
service latency over this window. The first block is faster because the queues
start empty. One block's wall time was far above the others (156 s against 23 s)
while its per-request latencies matched the rest; the stall sits in the unpaced
client path, four threads pushing through `adb forward`, not in the engine, so
the per-request rows are the ones to read.

Peak RSS for the hybrid server is 313 MB, measured on the device.

## Recognition quality

Twenty-four cabin commands, twelve Mandarin and twelve English, rendered with the
host speech synthesiser and labelled by their input text
(`tools/eval_texts.tsv`, `tools/make_eval_pack.sh`, `tools/eval_cer.py`). CER is
character level for Mandarin, WER is word level for English, both after removing
punctuation and spaces. One run per configuration over the same 24 files.

| Configuration | zh CER | en WER | clips with errors | clips with a special token |
|---|---:|---:|---:|---:|
| CPU, defaults | 36.79% | 6.58% | 12/24 | 9 |
| CPU, ConvNeXt in F32 | 38.68% | 6.58% | 12/24 | 0 |
| CPU, joiner on the host | 36.79% | 6.58% | 12/24 | 0 |
| HTP, polling | 36.79% | 6.58% | 12/24 | 0 |

What the table says:

- Quantising the ConvNeXt projections (H6) costs nothing measurable: 36.79%
  against 38.68% is one clip, in the direction of the quantised build.
- Moving the joiner to the device (H5) costs nothing measurable: identical
  character error on both backends.
- English is good on this set, 6.6% WER, five word errors in 76.
- Mandarin is dominated by the synthesiser voice, not by the engine: the errors
  are systematic confusions on one voice (请导航 heard as 请老黄, 把车内温度 heard
  as 老车里温度) that repeat across every configuration. This set cannot rank
  model variants for Mandarin; it can only show that a change does not make
  things worse.

Two artefacts worth knowing:

- The CPU device-joiner combination emitted a literal `<sos/eos>` token on 9 of
  24 clips; the host joiner and the HTP joiner emitted none. The evaluator strips
  the token and counts it separately. It appears only on clips that end in
  silence.
- Clips that end in silence used to gain a trailing character (轮) from the
  padding chunks. Flooring the BiasNorm sum removed the NaN that caused it and
  the trailing token mostly disappeared, but the decode of a silence tail is
  still the least stable part of the pipeline.

The noisy variant of this set is not usable yet: mixing a cabin recording at
equal weight made the speech unintelligible and every clip collapsed to the same
hypothesis. A level-calibrated mix is a separate piece of work.

Reproduce:

```sh
NOISE= examples/x-asr/tools/make_eval_pack.sh /tmp/eval-pack
python3 examples/x-asr/tools/eval_cer.py --manifest /tmp/eval-pack/manifest-clean.tsv \
    --model x-asr-zh-en-q8_0.gguf --device CPU --label "CPU default"
```

## B=1 chunk scheduler

Both arms use the same server binary, model, HTP device, four slots, and queue
limit of 64. Only `--chunk-scheduler` differs. Its worker alternates queue
admission with resumable requests and can yield after one encoder chunk plus
greedy work. It does not combine model inputs. For multichunk requests, stepping
can change how joiner calls are grouped across chunk boundaries.

The same binary includes the joiner bounds fix. Once a block of indices has
been consumed, a blank-only block advances to a fresh joiner call. An earlier
pilot predates this fix and is excluded from the results below. The run uses
the same GGUF and audio fixtures as the service baseline.

SM6650, three paced repetitions per arm and scenario, with one single-stream
warmup per arm. Latency is endpoint to final in milliseconds, in p50 / p95
order. A second ASR process was present and observed active on the device.

| Scenario | N per arm | Request scheduler | Chunk scheduler |
|---|---:|---:|---:|
| single | 3 | 168 / 170 | 173 / 327 |
| aligned4 | 12 | 733 / 1593 | 697 / 960 |
| mixed4 | 12 | 405 / 534 | 348 / 606 |

All 27 transcripts match. The maximum between-step wait was 90 ms for
aligned4 and 89 ms for mixed4, compared with a near-zero residual in the
request arm. This confirms interleaving. Median commit work stayed within
2 ms across arms in each scenario. The mixed tail regressed, the sample was
small and shared-device load varied, so the chunk scheduler remains opt-in.
Raw JSONL, timing logs, and hashes are in
`$HOME/Library/Caches/qwen3-asr-htp/xasr-multistream/chunk-yield.bSh3PG`.

Run after building and deploying the Android package with
[BUILD.md](BUILD.md). Point `XASR_REMOTE` at the package on the phone and
select the same host fixtures. The script starts and stops its own server on
an ADB-forwarded loopback port, records input and device binary hashes,
runs the three scenarios, and checks transcript parity.

```bash
export ANDROID_SERIAL=ADNQ6R5823007930
export XASR_REMOTE=/data/local/tmp/llama-x-asr
export XASR_ARMS='chunk request'
CACHE="$HOME/Library/Caches/qwen3-asr-htp"
bash examples/x-asr/tools/bench_chunk_scheduler.sh \
    build-android-xasr/bin/llama-x-asr-server "$CACHE/zh.wav" \
    "$CACHE/cabin-asr-probe/long-perf/zh-10.wav" \
    "$CACHE/cabin-asr-probe/long-perf/en-20.wav"
```

For staggered starts, prefix the same command with
`XASR_SCENARIOS=stagger-start4 XASR_REPS=1`. This uses four start offsets
of 0, 10,000, 17,100, and 17,400 ms and lengths of 20 s, 10 s, and two
short clips. The recorded single-round smoke test returned four matching
transcripts. Endpoint-to-final max was 1,001 ms in request mode and
1,098 ms in chunk mode. This is one round across four concurrent sessions.
Logs and the recorded planned offsets are in
`$HOME/Library/Caches/qwen3-asr-htp/xasr-multistream/stagger-start.HkmwgQ`.

The server log reports delay before first service as `queue_ms`, work spent on
the commit as `commit_ms`, and time from ingress to completion as
`elapsed_ms`. In the stepping arm, elapsed time also includes waits between
steps. These fields exclude the response socket write. The client records
endpoint-to-final over the ADB transport in both modes.

## Soak

Four sessions of mixed length (2.65 s, 2.65 s, 10 s, 20 s), unpaced, driven back
to back through the socket on the hybrid server (HTP x2 + CPU x2). Each round
creates four sessions, commits them and closes them, so this exercises session
creation, slot release and the encoding of 35 s of audio per round.

| Pass | Rounds | Sessions | endpoint p50 | p95 | p99 | max | peak RSS after | idle reaps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 160 | 234.5 | 425.9 | 778.6 | 810.3 | 316.8 MB | 0 |
| 2 | 20 | 80 | 232.6 | 452.4 | 727.1 | 878.5 | 316.8 MB | 0 |

Pass 1 starts at 294.7 MB and ends at 316.8 MB; pass 2 starts and ends at
316.8 MB, so the extra 22 MB is allocator retention for the per-stream buffers
(a 20 s session holds 1.3 MB of PCM and 0.7 MB of features), not a leak. The
per-stream p50 is balanced across the four slots (251, 253, 246, 168 ms), no
session was reaped by the idle timeout, which means every client closed cleanly,
and the server was still alive after 240 sessions.

Mixed lengths are much kinder than aligned endpoints: p95 here is 426 ms against
941 ms for four aligned 2.65 s clips, because the endpoints naturally stagger and
the second engine drains them.

Raw: `/tmp/soak-mixed.jsonl`, `/tmp/soak-mixed2.jsonl`.

## Fixed-lane HTP encoder batch probe

`--probe-encoder-batch` compares independent B=1 slots with one true
encoder graph for two or four distinct WAV inputs. The graph has a batch
axis on convolution, attention, masks, and recurrent caches. Linear layers
fold time and batch into the row dimension of a shared matrix multiplication.
Choose the batch size by passing two or four WAV files. Each lane uses its own
state and mask. The probe runs complete chunks up to the shortest lane, with
no endpoint padding, and reads raw encoder projections for comparison.

The measured unit is the wall time to execute one chunk for every lane,
including feature and mask upload, encoder execution, and projection readback.
Fbank, initialization, and the first two chunks are excluded. The reported
value is the upper middle sample after sorting the remaining chunk times
within a run. Lane assignment is fixed for this test. No cache movement,
joiner, server queue, or ADB response time is included.

SM6650 / v73, HTP completion polling enabled, command limit 2048. Values
below are rounded medians across three launches unless stated otherwise.

| Input setup | B | Chunks per lane | Serial B=1 group | Batched group | Change |
|---|---:|---:|---:|---:|---:|
| `zh-10.wav en-20.wav` | 2 | 20 | 168.2 ms | 132.5 ms | 21% less wall time |
| `zh-10.wav en-20.wav zh.wav en.wav` | 4 | 5 | 337.4 ms | 233.4 ms | 31% less wall time |
| Four 10–20 s WAVs, one launch | 4 | 20 | 338.7 ms | 233.8 ms | 31% less wall time |

Both batched graphs placed all 2,897 encoder compute nodes on HTP in one
scheduler split and emitted numerically identical encoder projections to
serial B=1 on the tested clips. Batch cache sizes in the graph were 8,297,472
and 16,594,944 bytes. The macOS CPU shape comparison showed about 3%
relative RMS difference on the short B=2 clips and did not pass the probe's
1% numerical gate. Treat this as an HTP result. The regular HTP B=1 CLI
returned the established short-clip transcripts.

Use the [Android build and deployment steps](BUILD.md). Build the
`llama-x-asr` target, push it to the deployed package, and run:

```sh
adb shell -T 'cd /data/local/tmp/llama-x-asr &&
    export LD_LIBRARY_PATH="$PWD/lib" ADSP_LIBRARY_PATH="$PWD/lib" &&
    export GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPBATCH=2048 &&
    bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 zh-10.wav en-20.wav --probe-encoder-batch'
```

For B=4 add `zh.wav en.wav` to the positional WAV list. For the long B=4
test use `zh-10.wav en-20.wav en-10.wav zh-20.wav`. The extra files came
from the local `cabin-asr-probe/long-perf` cache. Logs, input and device
library hashes, and the binary hash are in
`$HOME/Library/Caches/qwen3-asr-htp/xasr-multistream/encoder-batch.MCwTi2`.

The next service experiment needs to measure the remaining budget after
gathering and scattering arbitrary ready-lane state. The B=2 encoder saves
about 36 ms per pair before that cost. Start with ready-only batching and
allow B=1 immediately when no other lane is ready. Verify distinct sequence
positions, arbitrary lane reuse, and cancel/reset before measuring end-to-final
P95/P99 under staggered starts and aligned commits.

## Dynamic HTP encoder batching

`llama-x-asr-server --encoder-batch 2|4` batches ready encoder chunks. The
worker processes a bounded number of already queued requests before choosing
the largest available graph width (4, 2, or 1). There is no timer-based wait
to fill a batch. Every lane uploads features and a mask for that stream's
current position. Its 4,148,736 bytes of state are gathered and scattered over
the public tensor get/set interface, through a reusable host staging buffer.
The encoder graph executes on HTP. Transducer joiner and predictor work
continues per stream after the batched projection.

SM6650, HTP polling, command limit 2048, same GGUF and ARM64 server binary
across all three arms. The client is on the host via ADB forward. Two paced
rounds per scenario with four clients each, plus one single-stream warmup per
arm. Batch states and request handling were not changed between runs other
than the selected scheduler/width. A second ASR server process was present
on the phone. Treat these as smoke comparisons because of the sample count
and possible device contention.

| Workload | N/arm | B=1 chunk p50 / p95 | Up to B=2 p50 / p95 | Up to B=4 p50 / p95 |
|---|---:|---:|---:|---:|
| Aligned starts and endpoints | 8 | 698 / 963 ms | 608 / 873 ms | 605 / 808 ms |
| Staggered starts, clustered endpoints | 8 | 418 / 760 ms | 425 / 749 ms | 453 / 716 ms |
| Mixed short/long clips | 8 | 262 / 576 ms | 314 / 533 ms | 327 / 482 ms |

All 24 final transcripts per arm matched the B=1 chunk path. The up-to-B=2
arm made 61 true B=2 encoder dispatches. The up-to-B=4 arm made 50 B=2 and
10 B=4 dispatches. In B=2-only mode the median gather plus scatter cost was
4.91 ms, with median encoder compute 137.32 ms. Under the B=4 cap, width-4
gather plus scatter had median 13.37 ms and compute had median 237.13 ms.
The p95 improved on these three workloads, while the median rose for
staggered starts and mixed lengths. Keep batching opt-in pending repeated
arrival sequences under representative load.

Reproduce after the [Android build and deployment](BUILD.md):

```sh
export ANDROID_SERIAL=ADNQ6R5823007930
export XASR_REMOTE=/data/local/tmp/llama-x-asr
export XASR_ARMS='chunk batch2 batch4'
export XASR_SCENARIOS='aligned4,stagger-start4,mixed4'
export XASR_REPS=2
CACHE="$HOME/Library/Caches/qwen3-asr-htp"
bash examples/x-asr/tools/bench_chunk_scheduler.sh \
    build-android-xasr/bin/llama-x-asr-server "$CACHE/zh.wav" \
    "$CACHE/cabin-asr-probe/long-perf/zh-10.wav" \
    "$CACHE/cabin-asr-probe/long-perf/en-20.wav"
```

For the slot smoke test, start the server with `--streams 4 --encoder-batch 4`
and forward its port. Then pass the four local WAVs to:

```sh
python3 examples/x-asr/tools/test_batch_slots.py --port 18768 \
    zh.wav en.wav zh-10.wav en-20.wav
```

The SM6650 smoke log shows real width-2 and width-4 dispatches during a
four-client test with one large APPEND per client, one cancel, equal
transcripts for surviving streams, and successful reuse of the cancelled
slot. The host CPU CLI-to-server smoke test also matched on both short files.

A separate sherpa-onnx CPU control on the same phone, two threads and
unpaced, measured RTF 0.148 / 0.122 / 0.115 and median decoder-ready
chunk times 52.58 / 52.23 / 52.59 ms for the short Mandarin, 10 s Mandarin,
and 20 s English inputs. It uses the same INT8 export and 1,200 ms tail
as the [unified benchmark](#unified-benchmark-onnx-int8-gguf-cpu-gguf-htp).
This is an in-process single-stream reference. A same-arrival-sequence
ONNX multi-stream service control remains to be measured.

Read `xasr-batch` records in the server logs for width and phase times. The
run archive holds raw JSONL, placement logs, binary, model and input hashes,
source digest, build log, and the initial background-process sample under
`$HOME/Library/Caches/qwen3-asr-htp/xasr-multistream/service-real-batch.SaUcxi`.

## Adaptive CPU admission

`--adaptive-cpu` gates CPU overflow at OPEN. It reads per-core deltas from
`/proc/stat` for CPUs in the process affinity mask, sampled every 200 ms.
An affinity-eligible core is counted when at least 80% of its interval was
idle. CPU needs a free slot, at least `--threads` such cores, an empty
queue, and an idle worker. The HTP pressure signal is the age of its oldest
queued or active commit. HTP remains the default below 80 ms when a slot
is available. If HTP is full and CPU is unavailable, OPEN returns busy.
Existing streams retain their original engine and recurrent state.

The SM6650 smoke used batch cap 4, four HTP slots, two CPU slots, and two
CPU threads. An idle first OPEN used HTP. The following OPEN used CPU with
6 idle cores and a 513.7 ms HTP commit still outstanding, even though HTP
had free slots. Both requests completed, and the CPU short Mandarin
transcript matched its B=1 reference. With the process confined to one core,
four parked HTP streams filled the HTP slots. The next OPEN returned busy
with only one idle core. See the burst comparison below for end-to-end P95.

Reproduce from the repository root after deploying the updated server
using [BUILD.md](BUILD.md). From an ADB shell in the package directory
on the phone, start the normal-affinity server:

```sh
export LD_LIBRARY_PATH="$PWD/lib" ADSP_LIBRARY_PATH="$PWD/lib"
export GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPPOLL=1 XASR_TIMING=1
bin/llama-x-asr-server x-asr-zh-en-q8_0.gguf HTP0:0 --bind 127.0.0.1 \
    --port 18776 --streams 4 --cpu-streams 2 --encoder-batch 4 --adaptive-cpu \
    > adaptive-backlog.log 2>&1
```

In a separate host shell, set the path to your deployed package for the
log argument. The test sends its WAVs from the host.

```sh
export ANDROID_SERIAL=ADNQ6R5823007930
adb forward tcp:18776 tcp:18776
python3 examples/x-asr/tools/test_adaptive_cpu.py 18776 en-20.wav zh.wav \
    backlog adb /data/local/tmp/llama-x-asr/adaptive-backlog.log
```

For the no-headroom case, restart the same server with `taskset 1`
before `bin/llama-x-asr-server`, redirect to
`adaptive-no-headroom.log`, and run the test with mode `no-headroom`
and the corresponding log path. Android's mask syntax is hexadecimal
without the `0x` prefix. Use a permitted single-core mask if core 0 is
excluded. Remove the forwarding rule after testing with
`adb forward --remove tcp:18776`.

Raw test output, route logs, source hash and device hashes are archived in
`$HOME/Library/Caches/qwen3-asr-htp/xasr-multistream/adaptive-cpu.MqIT7W`.

### Finalized-segment burst comparison

`bench_adaptive_cpu.py` sends a 20 s English segment at t=0 and three
2.65 s Mandarin segments at t=550/600/650 ms. The client sends each segment
in one APPEND and immediately requests COMMIT. It measures from the request
arrival at the client to receipt of the final transcript, including OPEN and
TCP. This exercises delivery of complete VAD segments and creates HTP
pressure at admission. It does not represent paced audio or on-car noise.

Use the server launch above with `--threads 2` and export
`GGML_HEXAGON_OPBATCH=2048`. Keep `--streams 4 --encoder-batch 4` for every
arm. Restart the service for each arm. For default hybrid, retain
`--cpu-streams 2` and remove `--adaptive-cpu`. For HTP-only, remove both.
Run a one-repetition warmup for each start before collecting eight measured
repetitions. Use the `en-20.wav` and `zh.wav` fixtures from the Android
reproduction below. The run order on SM6650 was default hybrid, adaptive,
adaptive, default hybrid, then HTP-only.

```sh
python3 examples/x-asr/tools/bench_adaptive_cpu.py 18776 en-20.wav zh.wav --reps 1 > warmup.jsonl
python3 examples/x-asr/tools/bench_adaptive_cpu.py 18776 en-20.wav zh.wav --reps 8 > arm.jsonl
```

| SM6650 arm | Short segments | Short P50 (ms) | Short P95 (ms) | Short P99 (ms) | First short P95 (ms) |
|---|---:|---:|---:|---:|---:|
| HTP-only | 24 | 1977 | 2054 | 2055 | 2010 |
| Default hybrid | 48 | 965 | 1174 | 1176 | 781 |
| Adaptive CPU | 48 | 1585 | 1686 | 1757 | 551 |

Every measured adaptive burst sent only the first short segment to CPU.
Its OPEN arrived with roughly 500 ms of HTP commit age and 4 to 6 idle
cores. The next two short segments both went to HTP because the CPU worker
already had work. In the default hybrid arm, two short segments per measured
burst used CPU, up to its two-session capacity. Transcripts matched across
all arms at each arrival offset. Adaptive admission improved the first short
segment's P95 over default hybrid and increased the pooled short P95 by
512 ms. Keep the adaptive policy opt-in for this workload.

The raw client JSONL, per-arm server logs, device hashes, fixture WAVs,
and summary are in
`$HOME/Library/Caches/qwen3-asr-htp/xasr-multistream/adaptive-ab.P9xmgB`.

## Same-arrival comparison: ONNX CPU, HTP B=1, B=2, B=4

Every arm below is driven by the same client, over the same socket protocol,
with the same WAVs, 20 ms packets, 1,200 ms tail padding and client-side
endpoint-to-final timing. Per-request records, not single-stream RTF, are the
unit. The ONNX control is `tools/onnx_server.cpp`, a service wrapper with the
same protocol, one worker, commit priority and a drain after every APPEND, so
both engines do their encoder work while audio arrives and only the tail at
commit.

| Arm | Engine | Threads | Batch cap | Actual widths seen |
|---|---|---|---:|---|
| `onnx-cpu` | sherpa-onnx INT8, CPU | 2 | n/a | n/a |
| `htp-b1` | same GGUF, HTP polling | 2 | 1 | B=1 only |
| `htp-b2` | same GGUF, HTP polling | 2 | 2 | 91 x B=2 |
| `htp-b4` | same GGUF, HTP polling | 2 | 4 | 63 x B=2, 16 x B=4 |

Scenarios: one stream; four starting together; four with start offsets
0/250/500/750 ms; mixed lengths (2.65 s, 2.71 s, 10 s, 20 s); and a burst where
each VAD segment arrives as fast as the socket accepts. Three rounds per cell,
so N=3 for single and N=12 for the four-stream scenarios.

| Arm | single p50/p95 | aligned4 p50/p95/p99 | stagger4 p50/p95 | mixed4 p50/p95 | burst4 p50/p95 |
|---|---|---|---|---|---|
| `onnx-cpu` | 117.8 / 120.7 | 348.9 / 540.6 / 547.6 | 120.8 / 131.2 | 160.1 / 220.8 | 956.4 / 1596.3 |
| `htp-b1` | 171.1 / 176.9 | 598.4 / 946.7 / 950.5 | 266.8 / 370.5 | 255.3 / 455.3 | 1531.4 / 2447.4 |
| `htp-b2` | 166.3 / 174.6 | 593.8 / 856.0 / 856.9 | 275.4 / 372.2 | 273.8 / 485.6 | 1528.3 / 2439.4 |
| `htp-b4` | 169.1 / 173.1 | 573.9 / 811.1 / 827.7 | 305.7 / 342.7 | 300.0 / 483.4 | 1584.7 / 2443.7 |

Findings:

- The ONNX INT8 CPU control wins every scenario, by 1.4x on single, 1.7x on
  aligned four-way, 2.2x on staggered starts and 1.5x on the burst. Per chunk
  this is the same gap measured in the unified benchmark: about 52 ms per
  decoder-ready chunk for sherpa-onnx against 81 ms of DSP time for our encoder.
  Scheduling cannot close a per-chunk gap; this is the HTP graph work.
- Batching helps the case it was built for: aligned four-way p95 falls 946.7 to
  856.0 (B=2) and 811.1 (B=4), and staggered starts p95 falls 370.5 to 342.7 at
  B=4. It costs median time in mixed and burst loads (mixed p50 255 to 300,
  burst p50 1531 to 1585), because a wider graph waits for lanes that a serial
  schedule would have started.
- The burst case is the one that looks worst for HTP: all four segments arrive
  at once, so the tail is just 28 encoder chunks back to back, 2.44 s at 87 ms
  per chunk against 1.60 s at 52 ms for the CPU control. Batching does not
  recover it here: a burst has only four lanes and the run is dominated by the
  per-chunk cost.
- GGUF transcripts are identical across B=1, B=2 and B=4 on all 51 requests per
  arm, so the batched graphs are numerically equivalent on this workload.
- ONNX and GGUF disagree on 48 of 51 requests, and the difference is systematic
  rather than random: the INT8 export drops the leading 请 on the Mandarin
  fixtures (擎导航到上海虹桥机场) and reads the 20 s English clip as "Sarah. Set"
  where the GGUF reads "Sarah, said". Report those separately from latency.

Caveats: N=3 or 12 per cell, one device, client on the host through ADB forward,
one ONNX thread count (2) and one GGUF thread count. The client paced each packet
one interval early in this run, which biases every arm the same way by under
20 ms; the client is fixed for later runs and these numbers are otherwise
comparable.

Reproduce with one command. The driver starts one server per arm and round on
the phone, waits for its own `listening` line, forwards the port, replays the six
scenarios, records one JSON record per request, stops the server, and then writes
`summary.md` with the percentiles, the encoder batch widths, the placement and
the transcript parity:

```sh
python3 examples/x-asr/tools/bench_load_matrix.py --rounds 3 --interleave \
    --arms onnx-cpu,htp-b1,htp-b2,htp-b4 --out /tmp/xasr-batch1
```

`--interleave` runs round by round across arms instead of arm by arm, so slow
device drift hits every arm. The run directory holds `config.json` (git
revision, device, input hashes, binary hashes, per-arm command line and
environment), one `<arm>.jsonl` per arm, one `<arm>.log` per arm, and
`summary.md`. `tools/bench_parity.py --dir <run>` regenerates only the summary
from an existing run. `--htp-args` appends one experiment's flags to every HTP
arm, which is how the scheduling arms below were built.

## Same-arrival replay, 2026-11-17

Replay of the matrix above on the current build with the one-command driver, so
that every arm also carries its transcripts. Three rounds, interleaved across
arms, N=3 for `single` and N=12 for the four-session scenarios, no failed
request in 252. Batch cap 4, four HTP slots, ONNX two threads.

| Arm | single p50/p95 | aligned4 p50/p95 | stagger4 p50/p95 | mixed4 p50/p95 | burst4 p50/p95 | vadburst p50/p95 |
|---|---|---|---|---|---|---|
| `onnx-cpu` | 120.5 / 133.1 | 399.1 / 561.0 | 134.8 / 155.2 | 162.2 / 246.5 | 976.8 / 1509.8 | 941.1 / 2392.5 |
| `htp-b1` | 194.4 / 198.5 | 576.0 / 963.6 | 283.9 / 382.2 | 273.9 / 428.6 | 1701.6 / 2441.8 | 1506.9 / 3785.8 |
| `htp-b2` | 192.0 / 197.1 | 645.3 / 907.8 | 312.5 / 358.0 | 306.7 / 505.5 | 1535.6 / 2449.6 | 1600.4 / 5054.7 |
| `htp-b4` | 196.9 / 200.2 | 599.2 / 827.2 | 317.9 / 350.0 | 318.2 / 474.4 | 1513.4 / 2439.6 | 1928.4 / 5045.4 |

`vadburst` is the sixth scenario: one 20 s English recording delivered whole at
t=0 plus three 2.65 s Mandarin commands at 550/600/650 ms, each in a single
APPEND followed by COMMIT. It is the finalized-VAD-segment shape, and it is the
case where batching costs the most.

- The GGUF arms agree on all 63 requests per arm, so the batched graphs are
  still numerically identical on this workload (146 width-2 dispatches for
  `htp-b2`; 83 width-2 and 29 width-4 for `htp-b4`).
- ONNX and GGUF differ on 60 of 63, in two classes. 54 requests are the dropped
  leading 请 on the Mandarin fixtures. 6 requests are the 20 s English clip,
  where the ONNX detokeniser emits `navigate ... station. When ...` in lower
  case with periods and reads `Sarah. Set` for GGUF's `Sarah, said`. Report the
  two classes apart; neither is a latency result.
- Batching still moves the tail the wrong way on a burst. `vadburst` p95 rises
  from 3785.8 at B=1 to 5054.7 at B=2 and 5045.4 at B=4, because a wide graph
  waits for lanes that a serial schedule would already be running, and the 20 s
  lane keeps the width full. This is the failure mode the ETA router exists to
  avoid, and the reason batching stays opt-in.
- Aligned four-way is the one case where batching helps: p95 963.6 to 907.8 at
  B=2 and 827.2 at B=4.

Raw per-request JSONL, per-arm logs, configuration and summary:
`$HOME/Library/Caches/qwen3-asr-htp/xasr-load/2026-11-17-batch1/`.

## Routing under bulk VAD arrival

`--eta-router` replaces the least-loaded rule at OPEN with an estimated
completion order. Each engine scores `live_sessions + oldest_commit_age_ms /
300`; the smallest score wins, ties go to the device engine so model compute
stays on HTP, and the winner's slot is reserved before the request is queued.
Client sends one finalized 20 s English segment at t=0 and three 2.65 s Mandarin
segments at 550/600/650 ms (`tools/bench_adaptive_cpu.py`), which is the VAD
delivery pattern that made the old adaptive gate lose.

Batch cap 4, four HTP slots, two CPU slots, two CPU threads. Two passes with the
arms alternating, six repetitions each, same session and binary:

| Pass | Arm | short P50 | short P95 | short P99 |
|---|---|---:|---:|---:|
| 1 | default hybrid | 965 | 1621 | 1684 |
| 1 | eta router | 975 | 1159 | 1170 |
| 2 | default hybrid | 971 | 1621 | 1685 |
| 2 | eta router | 965 | 1168 | 1178 |

The router cuts short-segment P95 by 28 percent (1621 to 1163) with an unchanged
median, and the two passes agree to 1 percent. The gain comes from sending the
late short segments to the engine whose oldest commit is not already late,
instead of balancing queue length.

Paced traffic does not benefit: two rounds each, N=8 per cell.

| Scenario | default hybrid p50/p95 | eta router p50/p95 |
|---|---|---|
| single | 191.4 / 197.6 | 188.5 / 192.5 |
| aligned4 | 377.1 / 402.2 | 342.6 / 421.0 |
| stagger4 | 250.6 / 267.5 | 185.7 / 247.0 |
| mixed4 | 215.4 / 315.6 | 244.9 / 354.2 |

Staggered starts improve by 26 percent at the median, aligned four-way and mixed
lengths get 12 to 19 percent worse at P95. The router therefore stays opt-in:
it is the right policy for a burst of finalized VAD segments behind a long
recording, not for paced audio. Gating it on a long in-flight stream is the next
experiment, not a default change.

Earlier adaptive admission measured 1580/1678 short P50/P95 in the same session
as these arms, so the ETA rule is the better of the two opt-in policies for this
workload.

Reproduce:

```sh
# server on the phone
bin/llama-x-asr-server x-asr-zh-en-q8_0.gguf HTP0:0 --port 18765 \
    --encoder-batch 4 --streams 4 --cpu-streams 2 --eta-router
# host
python3 examples/x-asr/tools/bench_adaptive_cpu.py 18765 en-20.wav zh.wav --reps 8
```

Raw JSONL, per-arm logs and route decisions: `/tmp/xasr-router-1/`,
`/tmp/xasr-router-2/`, `/tmp/xasr-router-3/`.

## Routing on remaining work, 2026-11-17

The least-loaded rule compares queue lengths. A stream that is mid-append has
already left the queue, so under a burst of finalized VAD segments the rule
keeps choosing the engine that is already busy. `--route-resident` replaces the
queue with the audio an engine has accepted and not finished, times its own
measured ms per ms of audio, so a 20 s recording counts as roughly twenty
seconds of work rather than as one queue entry.

Resident audio is added when a request is enqueued and debited when it leaves
the worker for good, so a request parked between encoder chunks still counts.
`--eta-long-ms N` arms a gate for either router: beneath N ms of unprocessed
audio anywhere the server falls back to least-loaded. `--cpu-queue N` bounds the
CPU engine's queue separately, for the paced case only.

Both servers keep four HTP slots and two CPU slots with batch cap 4 unless the
arm name says otherwise:

```sh
# default hybrid, the control
bin/llama-x-asr-server x-asr-zh-en-q8_0.gguf HTP0:0 --port 18765 \
    --streams 4 --cpu-streams 2 --threads 2 --encoder-batch 4
# resident router behind a 3 s gate
... --route-resident --eta-long-ms 3000
```

Three rounds interleaved across arms, N=3 for `single` and N=12 for each
four-session scenario, 504 requests, no failure and no missing text:

| Arm | single p50/p95 | aligned4 | stagger4 | mixed4 | burst4 | vadburst |
|---|---|---|---|---|---|---|
| `onnx-cpu` | 126.7 / 133.6 | 370.5 / 582.5 | 134.4 / 151.9 | 175.1 / 259.6 | 1016.5 / 1577.4 | 944.3 / 2379.6 |
| `htp-b4` | 199.9 / 200.0 | 592.2 / 825.3 | 317.2 / 334.5 | 317.3 / 491.0 | 1680.9 / 2427.8 | 1929.7 / 5045.5 |
| `hybrid` | 196.7 / 198.1 | 344.7 / 415.4 | 253.4 / 276.8 | 260.2 / 402.4 | 856.8 / 1854.4 | 1011.5 / 4298.8 |
| `hybrid-cpuq` | 199.5 / 201.1 | 371.0 / 527.2 | 252.2 / 274.5 | 252.1 / 406.9 | 1039.0 / 1541.8 | 866.9 / 4315.0 |
| `hybrid-eta` | 203.8 / 204.2 | 378.5 / 405.9 | 202.5 / 243.1 | 226.9 / 283.9 | 864.9 / 1262.4 | 1012.6 / 4304.8 |
| `hybrid-resident` | 198.5 / 199.7 | 340.1 / 577.9 | 252.2 / 270.6 | 264.4 / 424.8 | 855.7 / 1525.8 | 1046.6 / 4295.9 |
| `hybrid-cpu3` | 201.4 / 202.0 | 394.3 / 571.6 | 222.5 / 271.1 | 253.9 / 440.1 | 854.8 / 1854.5 | 1016.9 / 4304.2 |
| `hybrid-resident-cpu3` | 188.2 / 201.6 | 373.8 / 401.5 | 250.5 / 271.7 | 227.4 / 282.9 | 926.0 / 1514.2 | 1289.5 / 3925.0 |

A pooled percentile mixes the 20 s recording with the 2.65 s commands, so the
same requests split by the audio they carried (`audio_ms` in the record). Short
p95, per arm: N=3 on `single`, 6 on `mixed4`, 9 on `vadburst`, 12 on the rest:

| Arm | single | aligned4 | stagger4 | mixed4 | burst4 | vadburst |
|---|---|---|---|---|---|---|
| `onnx-cpu` | 134 | 583 | 152 | 269 | 1577 | 1127 |
| `htp-b4` | 200 | 825 | 335 | 499 | 2428 | 1953 |
| `hybrid` | 198 | 415 | 277 | 428 | 1854 | 1124 |
| `hybrid-cpuq` | 201 | 527 | 275 | 417 | 1542 | 1129 |
| `hybrid-eta` | 204 | 406 | 243 | 257 | 1262 | 1125 |
| `hybrid-resident` | 200 | 578 | 271 | 426 | 1526 | 1121 |
| `hybrid-cpu3` | 202 | 572 | 271 | 451 | 1854 | 1129 |
| `hybrid-resident-cpu3` | 202 | 401 | 272 | 248 | 1514 | 1302 |

`burst4` and `mixed4` separated the arms, so the three candidates ran a second
pass, six rounds interleaved, N=24 per cell, on those two scenarios plus
`aligned4` and `vadburst`:

| Arm | aligned4 p50/p95 | mixed4 | burst4 | vadburst | short p50/p95/p99 (N=78) |
|---|---|---|---|---|---|
| `hybrid` | 377.9 / 456.0 | 237.4 / 360.7 | 924.0 / 1253.0 | 1015.0 / 4311.8 | 555 / 1230 / 1397 |
| `hybrid-eta` | 382.5 / 527.4 | 242.0 / 357.9 | 913.9 / 1262.5 | 1015.0 / 4299.8 | 549 / 1254 / 1264 |
| `hybrid-resident` | 381.3 / 527.4 | 252.9 / 416.6 | 855.7 / 1861.6 | 1076.2 / 4308.2 | 553 / 1260 / 1863 |

Findings:

- The default hybrid already beats HTP-only with no policy change: short p95
  825 to 415 on aligned four-way, 2428 to 1854 on the burst, 1953 to 1124 on
  the VAD burst. The CPU engine earns its two slots; the routing question is
  only about which of the four requests lands where.
- The VAD burst is already deterministic under the default rule. Three rounds,
  per-session values within 10 ms (`757 / 765 / 773`, `1122 / 1126 / 1115`,
  `902 / 901 / 908`). The 20 s APPEND and its COMMIT keep the HTP queue at one
  or two entries for the whole 650 ms window, so the queue-length rule splits
  the three commands 1 to 2 against HTP, every time. There is nothing left for
  a better router to fix in that cell: 1124 against 1121.
- The gate is exactly selective. It opened 9 of 64 routing decisions in the
  first pass and 18 of 97 in the second, which is precisely the three vadburst
  commands per round, and it never opened in `single`, `aligned4`, `stagger4`,
  `mixed4` or `burst4`. A gated policy therefore costs nothing on paced audio
  by construction, and that part of the design holds.
- Neither candidate beats the control, and the ordering flips between the two
  passes. In the three-round pass `aligned4` p95 was 415 for the control, 406
  for `hybrid-eta` and 578 for `hybrid-resident`; over six rounds it is 456,
  527 and 527. The candidates therefore lose 71 ms on that cell with more
  samples, and `hybrid-resident` is worse on `mixed4` p95 (416.6 against 360.7)
  and `vadburst` p50 (1076 against 1015). Keep the default and both routers
  opt-in.
- `--cpu-queue 1` cannot bind on this workload and its apparent win is a
  tie-break shift. A finalized segment is one APPEND followed by one COMMIT, so
  a CPU session never has more than one request outstanding; the cap only delays
  enqueues. The 866.9 against 1011.5 p50 comes from the three commands being
  assigned `s19/s21` instead of `s19/s20`, and the per-engine commit cost is the
  same in both arms (147 and 143 ms). Rejected.
- A third CPU slot makes the 20 s recording 9 percent faster and every command
  slower. `hybrid-resident-cpu3` sends all three commands to CPU, HTP keeps the
  recording to itself, and the recording's p95 falls 4296 to 3925 while the
  commands rise 1121 to 1302, because three sessions share two CPU threads. The
  CPU engine is thread-bound, not slot-bound. Rejected as a default.
- The `burst4` tail is bimodal, not noisy. The four lanes land at either
  `550 / 640 / 1070 / 1260` or `550 / 640 / 1250 / 1860`; the second regime is
  the two HTP lanes running one after the other instead of in one batch, and it
  costs 620 ms. Rounds in the second regime: 1 of 6 for `hybrid`, 3 of 6 for
  `hybrid-resident`, 0 of 6 for `hybrid-eta`. Routing only decides which lanes
  pair up; whether a pair shares a dispatch is the chunk scheduler's admission
  rule, and that is where the next change belongs.
- `--eta-router` is no longer the arm that was measured before. The age term
  used `oldest_active_commit_ms`, which the worker updated only under
  `--adaptive-cpu`, so it was always zero in every earlier router run. It is now
  maintained unconditionally, which roughly doubles the weight of the age term.
  The earlier 1621 to 1163 result therefore does not carry over, and today's
  `hybrid-eta` is within noise of the default.
- Host CPU is not the constraint. The server process burns 1462 ms of CPU per
  request at 0.78 of a core for `hybrid`, 1446 at 0.77 for `hybrid-resident`,
  and 979 at 0.53 for `htp-b4`; `hybrid-resident-cpu3` reaches 1408 at 0.81 and
  still loses on command latency.

Where the remaining gap actually is. Same file, same build, one stream, whole
file appended at once (`bin/llama-x-asr MODEL DEVICE en-20.wav zh-10.wav`):

| Engine | en-20 wall | RTF | zh-10 wall | RTF |
|---|---|---|---|---|
| our CPU engine | 3545 ms | 0.177 | 1766 ms | 0.177 |
| our HTP engine | 3832 ms | 0.192 | 2013 ms | 0.201 |
| sherpa-onnx INT8 CPU control | 2376 ms | 0.119 | - | - |

The device engine is 8 percent slower than our own CPU engine on the same file,
and both are about 1.6x slower than the ONNX control on the same CPU. So the
deficit is not the NPU: it is the graph, the layouts and the per-chunk work in
this implementation, and the ONNX control is the fair comparison to profile
against. That is what the next batch works on.

Raw per-request JSONL, per-arm logs, `cpu.json` and `summary.md`:
`$HOME/Library/Caches/qwen3-asr-htp/xasr-load/2026-11-17-batch2/` (eight arms,
three rounds, six scenarios) and `.../2026-11-17-batch2b/` (three arms, six
rounds, four scenarios).

Reproduce:

```sh
python3 examples/x-asr/tools/bench_load_matrix.py --rounds 3 --interleave \
    --arms onnx-cpu,htp-b4,hybrid,hybrid-cpuq,hybrid-eta,hybrid-resident,hybrid-cpu3,hybrid-resident-cpu3 \
    --out /tmp/xasr-batch2
python3 examples/x-asr/tools/bench_load_matrix.py --rounds 6 --interleave \
    --scenarios aligned4,mixed4,burst4,vadburst \
    --arms hybrid,hybrid-eta,hybrid-resident --out /tmp/xasr-batch2b
```

## Per-chunk budget and folded cache writes, 2026-11-17

Profiling the split first. `bin/llama-x-asr MODEL HTP0:0 en-20.wav` with
`XASR_TIMING=1` and `GGML_HEXAGON_PROFILE=1`, HTP0, batch 1, 43 chunks, 2,551
operators per chunk in two DSP batches (2,048 + 503):

| Group | ms/chunk | share | per call |
|---|---:|---:|---:|
| `MUL_MAT+ADD` (q8_0 projection) | 25.6 | 30.8% | 84.6 µs, 92.9 GMAC/s |
| `MUL_MAT` (small-K attention) | 10.5 | 12.6% | 104.3 µs, 25.6 GMAC/s |
| `CONT` | 7.5 | 9.0% | 585 calls, 8.1 µs fixed |
| `CONV_2D_DW` (ConvNeXt 7x7) | 5.6 | 6.7% | 1 call, 5.58 ms |
| `CPY` (streaming caches) | 5.4 | 6.5% | 115 calls, 46 µs fixed |
| `ADD`+`SCALE`+`SUB`+`MUL` | 8.8 | 10.6% | ~2,000 calls, 6.6–8.7 µs fixed |
| `IM2COL` (subsampling) | 3.8 | 4.5% | 3 calls, 1.26 ms |
| rest of the graph | 8.4 | 10.1% | |
| fbank, greedy, readback | 5.4 | 6.3% | outside `compute_ms` |

The per-operator fit `usec = fixed + slope * elements` is flat in size for every
elementwise operator: a `CONT` over 96 elements costs 9 µs, one over 65,536 costs
13 µs. So each of the ~2,000 small operators per chunk is ~7.5 µs of dispatch,
about 15 ms per chunk, 18 percent of it. This, not the NPU, is why the HTP engine
trails the ONNX control on the same CPU: ONNX pays 8.5 µs per kernel and we pay
7.5 µs per operator on the DSP plus a coarser split.

The cache write was the one place paying twice. Each of the 115 streaming caches
was built with `ggml_cont(last_cols(...))` and then copied into the cache, so the
same bytes moved twice and cost two graph nodes.

| Arm | graph nodes | DSP op ms (43 chunks) | `compute_ms` | `en-20` wall | RTF |
|---|---:|---:|---:|---:|---:|
| two-step (`XASR_CACHE_FOLD=0`) | 2,897 | 3,540 | 3,581.8 | 3,810.8 | 0.1905 |
| folded (default) | 2,782 | 3,175 | 3,210.9 | 3,440.8 | 0.1720 |

Six interleaved rounds, N=6 for `single` and N=24 per four-session scenario,
144 requests, 0 errors, 0 missing text, all six GGUF fixtures byte-identical:

| Scenario | folded p50 / p95 | two-step p50 / p95 |
|---|---|---|
| single | 173.9 / 177.7 | 198.2 / 200.1 |
| aligned4 | 356.9 / 381.9 | 356.5 / 588.3 |
| stagger4 | 206.5 / 233.2 | 253.1 / 272.2 |
| mixed4 | 251.3 / 392.4 | 256.3 / 460.5 |
| burst4 | 813.8 / 1134.9 | 838.3 / 1253.5 |
| vadburst | 1049.5 / 4283.0 | 1005.6 / 4283.9 |

The one adverse cell is `vadburst` p50 (+4.4 percent) with an unchanged p95; in
that scenario the 20 s recording holds the HTP queue for the whole arrival
window, so the sign of a small move there is not stable between passes. Pooled
short commands: p50 359 against 392, p95 1135 against 1246.

Reproduce:

```sh
python3 examples/x-asr/tools/bench_profile.py --wav en-20.wav --mode ops --env XASR_CACHE_FOLD=0
python3 examples/x-asr/tools/bench_load_matrix.py --rounds 6 --interleave \
    --arms hybrid,hybrid-nofold --out /tmp/xasr-batch3b
```

Raw JSONL, per-arm logs, `summary.md`, both operator profiles and the three-arm
run against the ONNX control: `$HOME/Library/Caches/qwen3-asr-htp/xasr-load/2026-11-17-batch3/`.

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
