# X-ASR on HTP

For commands, fixture hashes, expected placement, and measured results, see [BENCHMARK.md](BENCHMARK.md). Keep the model, fixtures, chunk geometry, and flush padding fixed when comparing graph revisions.

## Why the graph changed

The first SM6650 probe scheduled 41 encoder compute nodes on CPU and split the graph 43 ways. In a separate timing run, median encoder graph execution per chunk was about 173 ms in the mixed path and 100 ms on CPU. Graph construction and allocation took 30 ms vs. 7 ms, while output and state readback took 22 ms vs. 3 ms. Mixed graph execution includes synchronization and copies across backends. The figures describe the old probe and guide the order of the work below.

| Bottleneck | Change and rationale | What to verify |
|---|---|---|
| BiasNorm fell back to CPU in each layer | [`bias_norm`](xasr.cpp#L382) uses `SUM_ROWS` on squared centered values, applies `SQRT` to a contiguous 1-D view, and folds `sqrt(channel_count)` into the scale. For a channel count of `C`, `scale / sqrt(sum / C)` equals `scale * sqrt(C) / sqrt(sum)`. HTP supports the resulting graph and the learned bias and scale retain their roles. | Inspect the fallback list and compare encoder outputs and token IDs on CPU and HTP. The BiasNorm-only probe went from 43 splits to 3 and left the ConvNeXt matmul fallback. |
| ConvNeXt depthwise convolution lowered to a narrow matmul | [`build_chunk_graph`](xasr.cpp#L594) uses `ggml_conv_2d_dw_direct` for the 7×7 ConvNeXt depthwise convolution. Stride, frequency padding, and time history match the reference. | Compare the embedding output on the first chunk and on chunks with populated history. Confirm that the former convolution matmul disappears from the fallback list. |
| Each chunk rebuilt a graph and round-tripped the encoder state | [`prepare_graph`](xasr.cpp#L692) allocates the graph once and pins state to the target. The graph tail copies each state output into its input after the last reader. The old 4,148,736 bytes of state per direction per chunk are replaced by device-local copies. | Run the same clip twice on one context, then alternate Chinese and English. Confirm identical requests produce identical text and that a new stream has zero state. The current graph has one HTP scheduler split. |
| The same positional projections and convolution edge scales were built every chunk | [`project_positions` and `conv_edge_scale`](xasr.cpp#L459) calculate fixed tensors during setup. [`fixed_input`](xasr.cpp#L452) marks their storage persistent so the allocator cannot reuse it for an intermediate. Runtime masks remain position-dependent and are updated per chunk. | Scheduled compute nodes fell from 3,105 to 2,877. Validate both a short replay and a long replay, including chunks after the context window fills. Record initialization time separately if comparing cold-start performance. |
| Waiting for DSP completion still affected tail | With the same graph and limit of 2,048 ops, enabling `GGML_HEXAGON_OPPOLL=1` reduced median short-clip offline wall time from 1,011 to 815 ms. Raising the limit to 4,096 was slower in the tested sweep. | Compare polling on and off at the same batch limit and power mode. Check both end-to-final latency and CPU time under load. A scheduler split can contain several DSP submissions. |

### Isolating changes

The [historical probe files](BENCHMARK.md#historical-artifacts) record the initial graph and the BiasNorm-only rewrite. The checked-in example rejects encoder fallback. To test an old lowering in isolation, use the historical probe or an experimental copy of the example that records fallback placement and permits mixed execution. This keeps the production check intact.

For a fresh ablation, use one baseline build and vary one mechanism at a time. Restore the original `MEAN → SQRT` subgraph to test BiasNorm, or restore the composite `ggml_conv_2d_dw` call to test the convolution. For state residency, keep the graph cached and compare device-local copies with explicit cache read and upload. Measure graph rebuild and allocation in a separate run. To measure the constant-folding change, move only the relative-position projection and edge-scale expressions back into the per-chunk graph. Compare log placements before interpreting latency. The historical timing table spans several changes and cannot attribute the latency delta to any single graph rewrite.

## HTP experiments

Each entry follows the same shape: bottleneck and evidence, change, how to
reproduce, A/B result, and what it does not cover.

### H1. Bounded wait instead of spin for DSP completion (rejected, no code kept)

**Bottleneck.** With `GGML_HEXAGON_OPPOLL=1` the host spins in
`dspqueue_read` with a zero timeout. The tail is good (191 ms on the 2.65 s
clip) but the host burns 760 ms of CPU for a 3.85 s session. With
`GGML_HEXAGON_OPPOLL=0` the host blocks and uses 357 ms, but the tail is 273 ms.
A bounded timeout should give the polling latency at close to the blocking CPU
cost, so the hypothesis was that a few tens of microseconds of wait per attempt
would recover both.

**Change tried.** `GGML_HEXAGON_OPPOLL_US`: pass that value as the
`dspqueue_read` timeout instead of 0. Four lines in
`ggml/src/ggml-hexagon/ggml-hexagon.cpp` (the `flush_pending` read and the
environment parsing).

**Result.** Rejected. Any finite timeout behaves like the blocking path, even
20 us:

| Completion wait | zh 2.65 s flush | zh 2.65 s CPU | zh 10 s flush | zh 10 s CPU |
|---|---:|---:|---:|---:|
| block (OPPOLL 0) | 272.5 | 357 | 415.1 | 1204 |
| spin (OPPOLL 1, timeout 0) | 191.0 | 760 | 296.6 | 2541 |
| timeout 20 us | 271.9 | 384 | 419.8 | 1307 |
| timeout 50 us | 273.2 | 393 | 415.4 | 1353 |
| timeout 100 us | 269.9 | 387 | 414.1 | 1357 |
| timeout 250 us | 276.8 | 403 | 414.8 | 1313 |
| timeout 500 us | 271.8 | 386 | 411.7 | 1301 |

At 20 us the host already pays the blocking CPU cost and the blocking latency,
so there is no trade-off curve to pick from. The completion path costs about
7 ms per DSP batch when the host sleeps, which is larger than the whole
per-batch submit cost (0.2 ms). Reproduce with
`bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 zh.wav zh-10.wav --realtime`
and `GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPPOLL_US=<0|20|50|100|250|500>`;
the knob no longer exists because the result was negative.

**Consequence.** Polling is the only low-latency completion mode on this
device. Host CPU must be paid for, or hidden behind other useful work. That
moves the next HTP work to (a) fewer DSP ops per chunk and (b) overlapping a
second stream's host work with the first stream's DSP batch (Phase 3).

Raw log: `/tmp/poll-sweep-1.txt` on the development host; values above are
medians of two paced repetitions per cell.

### H2. Fuse the bias add into the HVX matmul (kept)

**Bottleneck.** The DSP profile of one chunk
(`GGML_HEXAGON_PROFILE=1`, 14 chunks over two replays) counts 9,730 ADD ops
for 14 chunks, 695 per chunk, and 120.6 ms of DSP time in 14 chunks, 8.6 ms per
chunk. Most of them are single-row biases added to a matmul result, for
example `640:24 x 640:1 -> 640:24`. On the DSP a 3,072-float add costs about
12 us, almost all of it dispatch: the same op moves 24 kB, which at the
measured 50 GB/s would take 0.5 us. The graph has 2,820 DSP ops per chunk at
about 86.6 ms, so op overhead is a large share of the chunk.

**Why HTP.** The backend already has a fused `MUL_MAT_ADD` op with a bias
parameter and the HVX kernel applies the bias over a row grid
(`hvx_tensor_add_f32_grid` in `htp/matmul-ops.c`). The fusion was refused on
HVX-only devices by one condition in
`ggml/src/ggml-hexagon/htp-fusion.cpp`:
`if (kparams.n_hmx == 0 && src1_nrows != 1) return false;`
so every matmul on a 24-frame activation lost its bias add. SM6650 has no HMX,
so this excluded the whole encoder from the fusion on exactly the device under
test.

**Change.** Allow the fusion on HVX when the added tensor is a broadcast bias,
that is when `src2` has a single row. The residual adds, which add a full
`C x T` tensor, keep the old path. Op shape, layout and dtype do not change:
`MUL_MAT q8_0 x f32 -> f32` plus `ADD f32 x f32 -> f32` becomes
`MUL_MAT+ADD q8_0 x f32 x f32 -> f32` at the same output shape, with the bias
read through `src2->nb[1]` inside the kernel.

**Reproduce.**

```sh
# Baseline commit: the parent of the H2 commit. Both arms use the same command.
adb shell -T "cd /data/local/tmp/llama.cpp && \
  export LD_LIBRARY_PATH=\$PWD/lib ADSP_LIBRARY_PATH=\$PWD/lib \
  GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_OPPOLL=1 \
  GGML_HEXAGON_POWER_MODE=burst && \
  bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 zh.wav zh-10.wav --realtime"

# DSP op census and per-batch time.
adb shell -T "cd /data/local/tmp/llama.cpp && ... GGML_HEXAGON_PROFILE=1 \
  bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 zh.wav zh.wav" 2> profile.log
python3 /tmp/agg_profile.py profile.log   # pipe-separated profile-op lines
```

**Result.** Fused bias adds: 289 fewer DSP ops per chunk. The second command
batch shrinks from 772 to 483 ops.

| Arm | DSP ops/chunk | batch 1 | batch 2 | DSP ms/chunk | zh tail (2 reps) | zh 10 s tail (2 reps) | text |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | 2,820 (2048+772) | 67.6 ms | 21.0 ms | 88.6 | 190.9, 191.1 | 295.9, 297.2 | reference |
| fused | 2,531 (2048+483) | 71.7 ms | 13.8 ms | 85.7 | 188.4, 188.7 | 293.8, 294.5 | identical |

DSP time per chunk falls 3.3% (88.6 to 85.7 ms) and the end-to-end tail falls
1.3% (about 2.5 ms on a two-chunk tail). The first batch becomes slower because
the fused ops now sit in it while the cheap ADDs are gone from the second; the
batch split is by op count, not by work, so the boundary moved. The measured
tail saving is about half what the profile predicts for two chunks because the
profile adds instrumentation and because the last chunk's completion overlaps
host work. Text is unchanged on both fixtures and on every paced run.

**What it does not cover.** Only the broadcast-bias case is enabled. HMX
devices keep the old guard; the fused solver has not been measured there.
Removing the guard entirely, so that a full residual tensor may fuse, was tried
and changed nothing: the DSP op count and the profile stayed at 5,684 ADD ops,
so the fused solver rejects the residual shape on its own. That path is closed
until the solver or the kernel is extended, and it should not be retried as a
one-line change.
Two-thirds of the remaining DSP time is still in small ops: 585 CONT, 115 CPY,
240 SCALE and 99 SOFTPLUS per chunk, 7.8 + 5.7 + 3.0 + 3.0 ms. Those are the
next HTP target, together with the two f32 ConvNeXt projections, which run at
6 to 10 GMAC/s against 40 GMAC/s for the q8_0 matmuls.

### H3. Shorter tail padding (rejected on these fixtures)

**Bottleneck.** `xasr_commit` appends 1,200 ms of zeros, which triggers two or
three extra encoder chunks. At 74.5 ms per chunk on CPU and 95 ms on HTP that is
most of the tail, so the obvious change is to append only what is needed.

**Hypothesis.** The last chunk of a 1,200 ms pad covers only silence, so it can
be dropped without changing the final text.

**Test.** `--pad-ms` accepts an explicit padding, and a phase-aware mode pads
only until the chunk that covers the last real frame can run:
`target = 48 * floor((frames - 1) / 48) + 62`.

| Padding | zh 2.65 s | zh 10 s |
|---|---|---|
| 1,200 ms (default) | 请导航到上海虹桥机场, 7 chunks | ...车内温度调到二十二, 23 chunks |
| phase-aware, 370 ms | 请导航到上海虹桥机 (场 lost), 6 chunks | ...车内温度调到 (二十二 lost), 21 chunks |
| none | 请导航到上海虹桥 (机场 lost), 5 chunks | ...车内温度 (二十二 lost), 20 chunks |

**Result.** Rejected. The extra silence chunk is load-bearing: it gives the
transducer the right context to emit the last token or two. The minimum safe
padding is phase dependent, because the chunk that covers the last real frame
is followed by a chunk that carries the trailing tokens. For the 2.65 s fixture
the real audio ends at frame 265, the covering chunk starts at 240, and the
emitting chunk starts at 288, so at least 850 ms is needed; 1,200 ms then costs
nothing extra on that fixture. On the 10 s fixture the same rule would save two
chunks, but only if 1,200 ms is not required to flush the last phrase, which the
table above contradicts.

**Next attempt.** Shorten the padding only after measuring the minimum per
endpoint phase over real VAD cuts, or emit the final text from a partial window
instead of padding. Both need a text-parity harness that runs many endpoint
phases; `tools/make_tail_fixtures.py` already generates the phase-shifted and
truncated clips for it.

### H4. Backend parameter sweep (no change kept)

**Question.** After H2 the chunk is 2,531 DSP ops. OPBATCH 2048 splits it into
2,048 + 483, so the host can hand the second batch over while the DSP is already
running the first. A larger limit removes the split; a smaller one adds more
overlap points but more submissions.

**Sweep.** Same build, `GGML_HEXAGON_OPPOLL=1`, two fixtures, two or three
paced repetitions per cell. Medians of the flush time in milliseconds.

| OPBATCH | burst | sustained_high_performance | high_performance | batches per chunk |
|---:|---:|---:|---:|---:|
| 512 | 187.9 (277.1 outlier) | | | 6 |
| 1,024 | 189.6 | | | 3 |
| 1,536 | 186.7 | | | 2 |
| 2,048 | 187.4 | 188.6 | 187.7 | 2 |
| 3,072 | 195.6 | | | 1 |
| 4,096 | 193.9 | 194.4 | | 1 |

**Result.** No change. 1,536 is within the 1-2 ms run-to-run spread of 2,048,
and 512 showed one 277 ms outlier, so it is less stable. 3,072 and 4,096 are
consistently 4-8 ms slower, which supports the overlap reading: with a single
batch the DSP cannot start until the host has finished enqueuing, while with
2,048 the first batch is already running. The power profile makes no measurable
difference on this SoC. The documented default stays OPBATCH 2048 and burst.

### H5. Joiner projection and argmax on the selected backend (kept, now the default)

**Bottleneck.** After H2 the encoder was fully on the DSP but the transducer
decoder was still on the host, and it was the second largest host cost.
Measured on the device with `GGML_HEXAGON_OPPOLL=1`, one 2.65 s utterance:
`greedy_ms` 53.6 ms over 84 encoder frames, 0.64 ms per frame. The host loop
reads `join.output_linear.weight` after `to_f32()` expanded it to 10.24 MB of
f32 per frame; the GGUF itself stores the same tensor as Q8_0, 2.56 MB. On the
host the same loop costs 3.40 ms per frame against 0.21 ms for onnxruntime
([ONNX-ANALYSIS.md](ONNX-ANALYSIS.md)), which is the weight traffic, not the
arithmetic.

**Why HTP.** The joiner is one 5000 x 512 by 512 x T matmul with no activation
between rows. The DSP runs q8_0 matmuls at about 40 GMAC/s, has an ARGMAX op,
and the weight tensor is already resident in the model buffer on that backend,
so there is nothing to move. Batching the 12 encoder frames of a chunk into one
call amortises the 2.56 MB weight read over 12 columns, and only the 24 KB
activation goes up and the 1 x 12 token ids come back.

**Change.** `prepare_joiner` builds a five-op graph on the same backend:
`tanh(add(act, pred))`, `mul_mat(weight, .)`, `add(bias)`, `argmax`. The bias add
is fused into the matmul by H2. Weight dtype and layout are untouched (Q8_0,
model buffer); the activation is f32, 512 x 12; the predictor state is f32,
512 x 1; the output is 1 x 12 int32. `greedy_run_dev` keeps the exact decoding
rule: frames are scanned in order, at most one symbol per frame, blank and unk
are never emitted, and an emitted token updates the predictor and discards the
remaining frames of the block, which are then recomputed with the new state.
The predictor update itself (`decoder_out`: embedding, grouped convolution,
relu, 512 x 512 projection) stays on the host; it costs 0.9 ms for the whole
2.65 s utterance, less than one extra DSP round trip would.

**Reproduce.** Baseline is commit `0456429d5`, experiment adds `xasr_joiner` and
`--joiner-host` to go back to the host loop.

```sh
# HTP, both arms, same binary.
adb shell -T "cd /data/local/tmp/llama.cpp && \
  export LD_LIBRARY_PATH=\$PWD/lib ADSP_LIBRARY_PATH=\$PWD/lib \
  GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_OPPOLL=1 \
  GGML_HEXAGON_POWER_MODE=burst XASR_TIMING=1 && \
  bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 zh.wav --realtime && \
  bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 zh.wav --realtime --joiner-host"
```

**Result.** HTP polling, two paced repetitions per cell, same binary and device.

| Arm | zh flush | zh 10 s flush | zh tail | zh 10 s tail | greedy | joiner on DSP | calls | predictor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| host joiner | 189.4, 189.7 | 298.0, 298.9 | 200.3, 207.3 | 299.3, 302.1 | 53.6 ms | 0 | 0 | 0 |
| device joiner | 179.7, 179.6 | 282.1, 282.0 | 191.8, 187.6 | 283.6, 284.4 | 20.9 ms | 18.4 ms | 16 | 0.9 ms |

Per DSP call 1.15 ms, against 0.64 ms per frame on the host times roughly five
frames per block, so about 2.8 times cheaper, plus the block is computed
speculatively instead of frame by frame. Endpoint to final improves 5.1 percent
on the 2.65 s clip and 5.4 percent on the 10 s clip. The four fixtures with the
device joiner: 178.4, 191.2, 277.9, 191.1 ms, all with the same text as the host
arm and as the CPU backend.

CPU backend, same device: 157.3, 179.1, 250.4, 172.6 ms against 165.6, 179.0,
255.0, 176.5 before. On the host CPU the effect is much larger, because the
host f32 loop is slower there: commit to final 87.1 to 46.7 ms on the 2.65 s
clip, `greedy_ms` 147.7 to 9.5 ms.

Over the full four-configuration matrix the per-call cost is 1.07 ms on
`gguf-htp-poll`, 1.74 ms on `gguf-htp-wait` and 3.56 ms on `gguf-cpu`, for 810
calls and 2,880 encoder frames each. The CPU backend call is a wash against the
host loop, because its quantised matmul for 12 columns is not faster than the
f32 loop in cache; the HTP call is about three times cheaper. The default is
uniform for both backends because the CPU arm is neutral and the graph keeps the
10.24 MB f32 expansion out of the process.

**What it does not cover.** Host CPU time is almost unchanged on the HTP arm
(752 to 764 ms for a 3.85 s session) because the polling spin dominates and the
host joiner was only about 15 ms per second of audio. The gain is latency, not
host CPU; the CPU gain has to come from Phase 3 overlap. The number of joiner
calls per chunk follows the emission rate, so a dense-token language or a
noisier corpus issues more calls; the worst case is one call per frame, which is
still cheaper per frame than the host loop. Text parity was checked on the four
fixtures only, plus repeat and language-alternation runs on one context.

### H6. Quantise the two ConvNeXt pointwise projections (kept, now the default)

**Bottleneck.** After H2 the DSP profile still had one operator pair far off the
fast path: `emb.cnx.pw1.weight` 128 x 384 and `emb.cnx.pw2.weight` 384 x 128,
both F32 in the GGUF, running as `MUL_MAT+ADD f32 x f32 x f32`. They cost
85.2 ms of DSP time over 14 chunks, 6.09 ms per chunk, for 44.8 M MAC, which is
7.4 GMAC/s. The quantised matmuls on the same device reach about 40 GMAC/s.

**Why HTP.** The weight is read once per call and the activation is 128 x 456,
so at this shape the matmul is weight-bandwidth bound. Q8_0 halves the weight
bytes and uses the tiled quantised kernel that already carries the rest of the
encoder. It also removes 92 KB of resident weight per stream, which matters when
several streams share the model.

**Change.** `quantize_into_q8` converts the two F32 tensors once at load with
`ggml_quantize_chunk` and uploads them as Q8_0 into a separate small context and
backend buffer, because the GGUF context is sized exactly for the file. The graph
selects the Q8_0 tensor when present. Shapes and layouts are unchanged
(`(128, 384)` and `(384, 128)`), the activation stays F32, and the bias add is
still fused by H2. `XASR_CNX_Q8=0` restores the F32 path for the A/B.

**Reproduce.**

```sh
adb shell -T "cd /data/local/tmp/llama.cpp && \
  export LD_LIBRARY_PATH=\$PWD/lib ADSP_LIBRARY_PATH=\$PWD/lib \
  GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_OPPOLL=1 && \
  XASR_CNX_Q8=0 bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 zh.wav --realtime && \
  XASR_CNX_Q8=1 bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 zh.wav --realtime"

# Operator census for either arm.
adb shell -T "... GGML_HEXAGON_PROFILE=1 bin/llama-x-asr ... zh.wav zh.wav" 2> profile.log
python3 /tmp/agg_profile.py profile.log
```

**Result.** DSP time per chunk 86.9 to 81.3 ms, minus 6.5 percent. The two
projections fall from 6.09 to 1.11 ms per chunk. The group
`MUL_MAT+ADD f32 x f32 x f32` disappears from the profile (28 ops, 85.2 ms over
14 chunks) and `MUL_MAT+ADD q8_0` grows by 28 ops and 15.6 ms.

| Arm | weights | DSP ms/chunk | projections ms/chunk | zh | en | zh 10 s | en 20 s |
|---|---|---:|---:|---:|---:|---:|---:|
| F32 | 196 KB | 86.9 | 6.09 | 179.9 | 193.4 | 282.6 | 195.1 |
| Q8_0 | 104 KB | 81.3 | 1.11 | 170.7 | 184.2 | 265.5 | 183.6 |

Endpoint to final drops 9.2, 9.2, 17.1 and 11.6 ms on the four fixtures, 5 to 6
percent, with identical text on all four and on both backends. CPU backend on
the device, three paced repetitions of the 2.65 s clip: 150.2 ms median before,
138.4 ms after, so the quantised weights do not hurt the CPU path.

**What it does not cover.** The subsampling convolutions still run as
`IM2COL f32 -> f16` plus `MUL_MAT f16 x f32` at 5.3 GMAC/s (3.8 ms per chunk for
the three of them). They cannot simply be quantised: their matmul input is the
F16 im2col buffer, and a Q8_0 weight against an F16 activation is not a path the
backend supports. That needs either an im2col that emits F32 or a mixed kernel.
Accuracy is checked on the four fixtures only; these weights are part of the
encoder front end, so a CER/WER sweep on real VAD cuts is still required before
this is called safe for deployment.

## Multi-stream

### M1. One encoder slot per stream, shared weights (kept)

**Bottleneck.** The context held one graph, one scheduler and one state set, so
it accepted one stream at a time. The weights are 168 MB and identical for every
stream, so running four contexts was not an option.

**Change.** `xasr_slot` now owns the encoder graph, its scheduler, its metadata
buffer and its 4,148,736 bytes of state. `xasr_context` holds a vector of slots
plus the shared model and the shared joiner graph. `xasr_stream_init` takes the
first free slot and `xasr_stream_free` releases it, so a stream that finished
can be replaced without rebuilding anything. `--streams N` sizes the pool and
`--streams N --concurrent` drives N aligned streams from the CLI.

**Reproduce.**

```sh
adb shell -T "cd /data/local/tmp/llama.cpp && \
  export LD_LIBRARY_PATH=\$PWD/lib ADSP_LIBRARY_PATH=\$PWD/lib \
  GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_OPPOLL=1 && \
  for n in 1 2 4; do bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 zh.wav \
    --streams \$n --concurrent --realtime; done"
```

**Result.** HTP polling, 2.65 s clip, one paced run per row.

| Streams | endpoint to final, in commit order (ms) | init | peak RSS |
|---:|---|---:|---:|
| 1 | 169.6 | 819 ms | 61.0 MB |
| 2 | 169.4, 340.5 | 831 ms | 78.0 MB |
| 4 | 170.0, 341.0, 512.2, 683.0 | 881 ms | 112.2 MB |

Text is identical on every stream, and a second and third wave in the same
process, including a Chinese to English switch, reproduce the same numbers, so
the slots are cleared on reuse. About 17 MB and 20 ms of init per extra slot.

With staggered endpoints the picture changes. Three streams end 1.5 s before the
fourth on a 10 s clip: 171.4, 343.2 and 514.9 ms for the three, and 269.6 ms for
the stream that kept recording, against 265 ms for that clip alone. Two streams
with a 3 s stagger: 171.9 and 268.9 ms. The continuing stream pays nothing for
the other commits because the DSP drains them while it is still receiving audio.

**What it does not cover.** The worker is single threaded and the streams are
serialized, so the four-way simultaneous endpoint costs one commit per stream,
683 ms for the last one. Interleaving chunks between streams does not change the
total work. The next step is encoder batching across streams, which needs a
stream dimension through the attention and its per-stream left context; the
feed-forward and convolution blocks can be widened without that.

### M2. One-way audio and commit priority in the server (kept)

**Bottleneck.** The first server version replied to every APPEND. At the endpoint
four clients send COMMIT within a few milliseconds of each other, and the worker
was a single FIFO: a commit queued behind another session's audio, and clients
waited on an APPEND reply that the worker could not produce while it was inside
someone else's commit. Measured single stream 185.9 ms and four aligned endpoints
440 to 961 ms, against 169.6 and 683 in-process.

**Change.** APPEND carries no reply. Audio is one way, backpressure is the
bounded queue plus the socket, and an APPEND the engine rejects closes the
connection instead of desynchronising the stream. The worker prefers a queued
COMMIT: it drains that session's earlier APPENDs in order, then runs the commit,
so per-session order holds and another session's audio cannot delay a finished
utterance.

**Result.** Single stream 167.3 ms p50 end to end against 169.6 in-process, and
aligned four-way p50 638.3, p95 941.4 over 48 samples. The engine time inside
those commits is p50 170.8 and p95 171.4, so the service adds no inference cost;
what remains is queueing.

**Not isolated.** Both changes landed together, so the numbers above are the pair.
The reason to keep the priority rule is the split that the instrumentation now
reports: queue wait p50 470.8 against commit p50 170.8 in the aligned case, and
the rule is what keeps a commit from being pushed back by other sessions' audio.

**What it does not cover.** FIFO among commits is still arrival order, there is
no batching and no admission control, and throughput under free-running load is
not measured. The queue is bounded at 64 requests, which is about 1.3 s of audio
per session at 20 ms packets, so a long stall can still push a client into
backpressure.

### M3. Hybrid engines, device plus CPU (kept)

**Bottleneck.** M2's instrumentation showed where the four-way tail lives: with
aligned endpoints the engine time inside a commit is p50 170.8 ms and the queue
wait is p50 470.8, p95 775.2. That queue is not protocol overhead, it is the
single worker spending time inside another stream's commit while the eight CPU
cores are idle.

**Hypothesis.** The device DSP and the CPU are separate processors. If two
streams commit on the DSP while two commit on the CPU, the four-way tail should
drop toward half even though each commit is no faster.

**Change.** The server now holds one engine per backend, each with its own worker
thread, request queue and stream slots. OPEN is pinned to the least loaded engine
that has a free slot; everything after that is dispatched to that engine's queue.
The engine itself is unchanged, so the CPU engine loads its own copy of the
weights.

**Reproduce.**

```sh
# device, both configurations back to back
adb shell -T "cd /data/local/tmp/llama.cpp && export LD_LIBRARY_PATH=\$PWD/lib \
  ADSP_LIBRARY_PATH=\$PWD/lib GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 \
  GGML_HEXAGON_OPPOLL=1 && bin/llama-x-asr-server x-asr-zh-en-q8_0.gguf HTP0:0 \
  --port 18765 --streams 4 --cpu-streams 2"
python3 examples/x-asr/tools/bench_service.py --port 18765 --reps 8 \
  --short zh.wav --long zh-10.wav --long2 en-20.wav --scenarios aligned4,single
```

**Result.** Aligned four-way, 8 repetitions (N=32): p50 636.5 to 302.2 ms, p95
944.0 to 433.0 ms. Single stream 166.5 to 165.0 ms. Peak RSS 112 to 313 MB
because the CPU engine holds a second weight copy. The server log shows 24
commits on the device engine and 16 on the CPU engine in the hybrid run, and the
transcripts are identical across engines.

**What it does not cover.** The 2+2 split is fixed. It is the natural split for
four equal streams because both engines then carry two commits, but a burst of
short utterances, a long recording plus three commands, or a run where the CPU is
already busy with feature extraction will not balance. Admission control that
picks the engine per arrival, and a split derived from the observed arrival
pattern, are the next step. The CPU engine also steals cycles from the predictor
and the feature extractor that run inside the device engine's commits, which the
numbers above already include.

### H7. OpenCL GPU as a third engine (rejected)

**Idea.** The device exposes an Adreno 810 through the OpenCL backend, and the
hybrid result showed that a second processor nearly doubles capacity. A third
engine could take a third of the commits.

**Test.** `llama-x-asr ... GPUOpenCL zh.wav` on the device. The backend loads
about 300 kernels on first use; with a 20 s timeout the process was still in
kernel loading and never reached the encoder. The device names `OpenCL`, `GPU`
and `MTL0` do not resolve, so `GPUOpenCL` is the backend's device name and the
blocker is compilation, not discovery.

**Why it is not worth pursuing here.** Even with a warm kernel cache the encoder
would have to place all 2,877 nodes on OpenCL, because the engine rejects
fallback: the graph contains a depthwise convolution, `ssm_conv`, argmax and a
large number of tiny elementwise ops whose per-dispatch cost on a GPU is worse
than on the DSP. The two-engine split already reaches 8.3 times real time, and
the third engine would have to beat a 1.0 s saturated tail to matter.

Recorded so the next attempt starts from the kernel cache and the op coverage
question instead of rediscovering them.

### H7. Fold the streaming-cache write into one copy (kept, now the default)

**Where the 83 ms per chunk goes.** `XASR_TIMING=1` and
`GGML_HEXAGON_PROFILE=1` on `en-20.wav`, HTP0, batch 1, 43 chunks, 2,551
operators per chunk in two DSP batches (2,048 + 503). Operator time is the DSP
body only; it sums to 3,540 ms against 3,579 ms of `compute_ms`, so the split
below carries the batch boundaries and the host submit as a small residual:

| Group | ms/chunk | share |
|---|---:|---:|
| `MUL_MAT+ADD` (q8_0 projection, 92.9 GMAC/s) | 25.6 | 30.8% |
| `MUL_MAT` (small-K attention, 25.6 GMAC/s) | 10.5 | 12.6% |
| `CONT` (layout, 585 calls) | 7.5 | 9.0% |
| `CONV_2D_DW` (ConvNeXt 7x7, 1 call, 5.58 ms) | 5.6 | 6.7% |
| `CPY` (115 cache writes, 46 µs each) | 5.4 | 6.5% |
| `ADD` / `SCALE` / `SUB` / `MUL` (2,000 calls) | 8.8 | 10.6% |
| `IM2COL` (3 subsampling convs) | 3.8 | 4.5% |
| `SOFTPLUS`, `SSM_CONV`, `SOFT_MAX`, `CONCAT`, `PAD`, rest | 8.4 | 10.1% |
| outside the graph (fbank, greedy, readback) | 5.4 | 6.3% |

The per-call floor is the point. Fitting `usec = a + b * elements` per operator
over all 110k records gives `a` between 6.3 and 8.7 µs for every elementwise
operator, and the fit is flat in size: a `CONT` over 96 elements costs 9 µs and
one over 65,536 elements costs 13 µs. Two thousand small operators per chunk at
~7.5 µs is 15 ms, 18 percent of the chunk, spent on dispatch rather than
arithmetic. `CPY` is the exception at 46 µs; the extra 38 µs sit in the DMA
hand-off, not in the bytes (`b` is 0.14 ns/element, 28 GB/s).

**Bottleneck.** Every streaming cache was written twice. `last_cols()` builds
the window with `ggml_cont`, materialising 8–131 KB into its own graph tensor,
and the tail of `build_chunk_graph` then copies that tensor into the cache with
`ggml_cpy`. 115 caches per chunk: `emb_cache` plus 19 layers x 6.

**Change.** `cache_write()` records the strided window directly and
`add_cache()` copies from it, so the cache write is one `ggml_cpy` whose source
is a `ggml_view_3d`. The downstream copy already handles a strided source
(`HTP_COPY_KERNEL_RESHAPE`), the value never leaves the graph, and the graph
loses 115 nodes. `XASR_CACHE_FOLD=0` restores the two-step path for the A/B.

**Reproduce.**

```sh
# Single stream, both arms, same binary and same device.
adb shell -T "cd /data/local/tmp/llama.cpp && \
  export LD_LIBRARY_PATH=\$PWD/lib ADSP_LIBRARY_PATH=\$PWD/lib \
  GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_OPPOLL=1 XASR_TIMING=1 && \
  XASR_CACHE_FOLD=0 bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 en-20.wav && \
  XASR_CACHE_FOLD=1 bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 en-20.wav"

# Stage and operator budget for either arm.
python3 examples/x-asr/tools/bench_profile.py --wav en-20.wav --mode ops --env XASR_CACHE_FOLD=0

# Six interleaved rounds, both arms, the six arrival scenarios.
python3 examples/x-asr/tools/bench_load_matrix.py --rounds 6 --interleave \
  --arms hybrid,hybrid-nofold --out /tmp/xasr-batch3b
```

**Result.** Operator count 110,373 to 105,428 over the run, exactly the 4,945
caches removed; `CONT` drops 25,155 to 20,210 calls and 324 to 221 ms. DSP
operator time 3,540 to 3,175 ms. Single stream on `en-20.wav`, adjacent runs,
only the environment differs: `compute_ms` 3,581.8 to 3,210.9 (−10.4 percent),
wall 3,810.8 to 3,440.8 ms, RTF 0.1905 to 0.1720. `zh-10.wav` 1,911.9 to
1,714.1 ms of compute.

Six interleaved rounds on the phone, N=6 for `single` and N=24 per four-session
scenario, 0 errors, 0 missing text:

| Scenario | fold p50 / p95 | two-step p50 / p95 |
|---|---|---|
| single | 173.9 / 177.7 | 198.2 / 200.1 |
| aligned4 | 356.9 / 381.9 | 356.5 / 588.3 |
| stagger4 | 206.5 / 233.2 | 253.1 / 272.2 |
| mixed4 | 251.3 / 392.4 | 256.3 / 460.5 |
| burst4 | 813.8 / 1134.9 | 838.3 / 1253.5 |
| vadburst | 1049.5 / 4283.0 | 1005.6 / 4283.9 |

Pooled over the 108 short commands: p50 359 against 392, p95 1135 against 1246.
Transcripts are byte-identical on all four fixtures, on the repeated and
interleaved requests, and on the CPU backend, where the graph also drops from
2,897 to 2,782 nodes. Cancel, surviving-stream parity and slot reuse pass.

**What it does not cover.** The one adverse cell is `vadburst` p50, 1049.5
against 1005.6, and its p95 is identical. In that scenario the 20 s recording
holds the HTP queue for the whole arrival window, so a faster encoder changes
when the three commands reach the head of the queue rather than how long they
wait; the sign of a 4 percent move there is not stable across passes. Two things
stay untouched: the other 2,436 operators per chunk still pay ~7.5 µs each, and
`CONV_2D_DW` at 5.58 ms plus `IM2COL` at 3.8 ms remain the two largest single
non-matmul items. The 46 µs `CPY` floor also remains for the writes that do not
fold, and it is the next target for the same reason: 38 µs of it is hand-off
overhead rather than bandwidth.

One caveat on reading the operator table: `ADD`, `SCALE` and `MUL_MAT+ADD` also
got 20 to 25 percent cheaper per call after this change while their operand
count and bytes did not move. Removing 4,945 intermediate tensors lowers L2
pressure for everything that follows, so a single profile pass can attribute a
global cache effect to individual operators. Use the pooled service numbers for
the decision and the operator table only for ranking.

### H8. Where the remaining operators go (measured, nothing kept)

The operator table ranks by op type. That is the wrong axis for the next cut,
because the same op type covers both a 3,072-element bias add and a 157 KB
transpose copy. `XASR_DUMP_GRAPH=1` prints the graph as `op, name, ne, bytes`
per node, so the count and the bytes can be attributed to a node family:

```sh
XASR_DUMP_GRAPH=1 build-x-asr/bin/llama-x-asr x-asr-zh-en-q8_0.gguf CPU clip.wav 2> nodes.log
```

On the B=1 graph: 3,738 nodes, of which 2,782 are executed (the scheduler skips
`VIEW`, `RESHAPE`, `PERMUTE` and `TRANSPOSE`), i.e. the count the profile sees.

| family | nodes | MB written | where it comes from |
|---|---:|---:|---|
| `ADD` | 695 | 12.4 | bias adds from `lin`, residual adds, mask and position adds |
| `CONT` | 470 | 10.5 | two depthwise transposes per layer (157 KB each, ~57% of these bytes) plus the attention permutes |
| `MUL_MAT` | 425 | 8.1 | linear layers and the four small-K attention matmuls |
| `SCALE` | 240 | 8.0 | 198 of them are the two constants inside `swoosh` |
| `SUB` | 162 | 4.6 | `swoosh` output, `bias_norm`, `bypass` |
| `MUL` | 157 | 1.9 | `bypass`, gates |
| `UNARY` | 156 | 4.3 | `softplus`, `sigmoid`, `sqrt` |
| `CONCAT` | 117 | 5.3 | the streaming cache window |
| `CPY` | 115 | 4.1 | one cache write per cache tensor per chunk |
| `SSM_CONV` | 76 | 0.9 | depthwise convolution on the transposed tensor |

At ~7 µs of dispatch per executed node, 2,449 nodes per chunk are roughly 17 ms
of a 77 ms chunk. Three batches of graph work removed the bookkeeping nodes
(4,945 of them in H7); what is left is arithmetic and cache traffic, so the two
candidate cuts left are small and were not taken:

- **Fold the `swoosh` offset into the preceding bias.** `swoosh(x, off, c)` spends
  one `SCALE` on `x - off`, but its input is always a linear output whose bias is
  added anyway, so `-off` can live in that bias. 99 nodes per chunk, about 0.7 ms,
  i.e. under 1%. It costs a shifted copy of 99 small bias tensors in the model.
- **Replace the depthwise `transpose + CONT + SSM_CONV` with K tap MACs.** K is 3
  to 5, so 2K-1 elementwise ops replace a 157 KB copy plus a kernel call: about
  2% of the chunk, and not bit-exact because the accumulation order changes.

Both are below the run-to-run spread of the four-stream service numbers, so the
graph is treated as at its floor for this chunk size. The next double-digit
change has to alter the arithmetic rather than the bookkeeping: a wider chunk
that amortises the per-node fixed cost over more audio, or a fused
`scale_bias`/`softplus` on the DSP side so a `swoosh` costs one node instead of
four.

Two diagnostics came out of the same work and stay: `XASR_DEBUG_LOGITS=1` prints
the five best joiner candidates at every emitted token (raw logits, so a gap can
be compared against a bias), and `XASR_DEBUG_FIRST=N` traces the first N frames
even when the winner is blank, which is how an early-emission theory was ruled
out for the English `Set` error. Both require the host joiner (`--joiner-host`
or the CLI default); the device path returns only the argmax.

### H9. Batching stops paying at a pair, and the tail needs the second engine

**Bottleneck.** H8 left the graph at its dispatch floor, so a smaller four-way
tail had to come from how the work is placed rather than from the graph.
`--encoder-batch` arrived after M3 was measured, which reopened the question: is
one device engine enough for four streams once lanes can share a step?

**What a chunk costs at each batch width.** `--probe-encoder-batch` runs the same
chunks lane by lane and then as one batched step, and reports the median of
chunks 3 and later, so the numbers exclude the first chunk's cold caches.

```sh
adb shell -T "cd /data/local/tmp/llama.cpp && export LD_LIBRARY_PATH=\$PWD/lib \
  ADSP_LIBRARY_PATH=\$PWD/lib GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 \
  GGML_HEXAGON_OPPOLL=1 && bin/llama-x-asr x-asr-zh-en-q8_0.gguf HTP0:0 \
  zh-10.wav en-20.wav --probe-encoder-batch"
```

| lanes | serial step | per lane | batched step | per lane |
|---:|---:|---:|---:|---:|
| 2 | 149.3 ms | 74.6 ms | 123.3 ms | 61.6 ms |
| 4 | 297.4 ms | 74.4 ms | 220.5 ms | 55.1 ms |

The three widths (1 from the serial arm, 2 and 4 from the batched one) fit
`step = 25.5 ms + 48.9 ms x lanes`, and that model predicts the four-lane step at
221.1 ms against 220.5 ms measured. A step therefore has a
fixed 25.5 ms (2,782 executed nodes at about 9 us, plus the two submissions the
2,048-op batch limit forces) and 48.9 ms of arithmetic per lane. Batching
removes the fixed part and nothing else: with dispatch free, four lanes would
still need 195 ms, so the measured 220.5 ms is 89% of batching's own ceiling and
the 1.35x against serial is all this width can give.

That answers the question the tail was waiting on. Four lanes on the device cost
2.9 single streams, not one, because the arithmetic does not shrink; the second
engine is not an optimisation on top of batching, it is the only thing left.

**The placement rule had to change first.** `reserve_least_loaded` compared
`queue.size()`, and a worker drains that queue before the next OPEN is routed, so
each simultaneous arrival saw an empty queue on an engine that was already
running a request. Four arrivals therefore piled onto the wider engine: the
default four-device-plus-two-CPU deployment produced 16 device and 9 CPU
placements over five rounds and behaved like 3+1 at 1,324 ms p95. Placement now
ranks by what a lane costs, which is the table above written as a rule: complete
an odd device lane (0), take an idle engine (1), start a new device step (2), then
fewer live lanes, then the shorter queue. A pair is filled before a third lane
starts a step, and a second CPU lane is parallel rather than queued because it
has its own slot and threads.

**Result.** Five rounds per arm, arm order rotated per round, one invocation per
arm and round with a warm-up clip first, four whole segments of 2.2 to 2.7 s
committed at the same instant, N=20 per arm, transcripts identical across every
arm (`tools/bench_engine_split.py`):

| arm | p50 | p95 | p99 |
|---|---:|---:|---:|
| four device slots, batch 4 | 1,495 | 1,614 | 1,615 |
| four device slots, no batch | 1,358 | 1,942 | 1,944 |
| four device + one CPU slot | 1,052 | 1,328 | 1,330 |
| three device + one CPU slot | 979 | 1,372 | 1,374 |
| **default: four device + two CPU slots** | **892** | **1,022** | 1,069 |
| two device + two CPU slots (explicit) | 880 | 995 | 999 |

Two things in that table are not obvious. Batching lowers the tail 17% (1,942 to
1,614) while raising the median 10% (1,358 to 1,495): a batch step advances four
lanes together, so a lane's finish is gated by the slowest member, but the unlucky
lane no longer waits three serial steps. And the split is decided by how many CPU
slots exist, not by the batch width: one CPU slot gives 3+1 and recovers 15% of
the four-device tail, two slots give 2+2 and recover 37%, or 38% when the device
engine is limited to two slots so the pair cannot be broken. Before the placement
fix the default deployment measured 1,324 ms p95 on the same harness, so the rule
is worth 23% on its own.

**What it does not cover.** These are aligned arrivals, the worst case for
`endpoint -> final`, not the common one; staggered arrival is priced in M3 and in
the `vadburst` scenario of the load matrix. Five or more streams would fill both
engines' pairs and the rule then spills back to the device. `--threads 3` and
`--threads 4` on the CPU engine did not complete within the client's 30 s timeout
while `--threads 2` finished in about one second; the phone showed 800 MB free and
2.1 GB of swap in use at the time, so two threads is the most that has been shown
to work rather than a measured optimum.

## Stage baseline after six batches

Six measured batches between 2026-11-16 and 2026-11-17 fixed the comparison
harness, then moved the service from "one device engine, unknown cost" to a
two-engine layout with a placement rule. This section states where that left
it, so the next change is judged against a baseline rather than against whichever
number was quoted last. Every figure below is on SM6650, with the fixtures and
hashes in [BENCHMARK.md](BENCHMARK.md#setup), 20 ms packets, and 1,200 ms of
tail padding.

**Default deployment.** Four HTP lanes with batching, two CPU lanes with two
threads each, cost-ranked placement. This is the configuration the table at the
end of H9 was measured with.

```sh
export LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib \
       GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_OPPOLL=1
bin/llama-x-asr-server x-asr-zh-en-q8_0.gguf HTP0:0 --port 8765 \
    --streams 4 --encoder-batch 4 --cpu-streams 2 --threads 2
```

| Measure | Harness | N | p50 | p95 |
|---|---|---:|---:|---:|
| one stream, in process | `bench_profile --mode timing`, `en-20`, 43 chunks | 1 | 3,440.8 ms wall, RTF 0.172 | |
| one stream, through the socket | `bench_load_matrix`, `single` | 6 | 173.9 | 177.7 |
| two aligned streams | CLI `--streams 2` | 10 requests | 968 / 977 | |
| four aligned segments, 2.2–2.7 s | `bench_engine_split`, default arm | 20 | 892 | 1,022 |
| one stream, whole clip, no queue | `bench_vehicle`, Tingting pack | 96 | 546 | 701 |

The first and last rows are not comparable: the CLI runs one process with no
service queue, the socket rows include the forward and the queue, and the
four-way row commits four segments at the same instant, which is the worst case
for `endpoint -> final`, not the common one.

### What each batch settled

| Batch | Commits | Result |
|---|---|---|
| 1. Same-load comparison | `231249726` | B=1/2/4 return byte-identical text on every fixture (63/63). Batching only pays when endpoints align: `aligned4` p95 964 → 908 → 827 ms, `vadburst` 3,786 → 5,055 → 5,045 ms. |
| 2. Routing and the bulk control | `1f0141452`, `4629a229f` | The default hybrid already exhausts the CPU engine's worth: short-command p95 825 → 415 ms. Bulk RTF is 0.177 on our CPU and 0.192 on HTP against 0.119 for the sherpa-onnx INT8 CPU control, so the gap is this implementation's graph, not the NPU. |
| 3. Cache fold | `87b021a64` | One copy instead of two per streaming cache: 4,945 fewer nodes, `compute_ms` −10.4%, RTF 0.1905 → 0.1720, single-stream p50 198.2 → 173.9 ms. Each remaining operator costs a flat 6.3–8.7 µs regardless of size. |
| 4. Cabin command fields | `2f259dffd`, `c43006ac5` | The same rows read 8.43% CER and 68.8% executable commands; a phrase bias repairs nouns and reaches 75.0% but not the verb. One speaker's voice does not transfer at all (34.4%). |
| 5. Phrase-bias ceiling | `eca64cc26`, `85e98575a` | A word-initial token can win at most 0.9 logit of bias, while the English `Set` misses need 1.76–6.42, so no legal entry fixes them. Bias is per stream: 12/12 identical under four concurrent sessions. |
| 6. Placement and engine split | `782f62ef1`, `f185e1b9f` | Ranking a lane by what it costs instead of by a queue the worker already drained: four-way p95 1,324 → 1,022 ms. Batching is worth 1.35x at four lanes and no more; the second engine is what carries the tail. |

### Where the limits now are

- **Graph dispatch is the floor at this chunk size.** 2,449 executed operators
  per chunk at ~7 µs is 17 ms of a 77 ms chunk. The two remaining bookkeeping
  candidates (`swoosh` offset folding, depthwise tap MACs) are each under 2%,
  below the four-stream spread, so the next double-digit change has to alter the
  arithmetic: a wider chunk, or a fused `scale_bias`/`softplus` on the DSP side.
- **Batching is worth 1.35x, not four.** `step = 25.5 + 48.9 x lanes` predicts
  the four-lane step within 0.3%. With dispatch free four lanes still cost
  195 ms, so 220.5 ms measured is 89% of batching's own ceiling. Batching
  amortises the fixed part and nothing else.
- **The four-way tail is set by the CPU lane.** At two lanes each, the device
  spends 61.6 ms per chunk per lane and the CPU engine about 135 ms, so 2+2
  finishes when the CPU pair does. Everything above that in the p95 is queueing.
- **A phrase bias cannot move a first-token decision that far.** The head token
  receives `0.3 x gain` and `gain <= 3`, so the ceiling is 0.9 logit. Fixes for
  the English failures have to be a constrained decoder or a different acoustic
  model.
- **The CPU engine is memory-bound before it is thread-bound.** `--threads 3`
  and `--threads 4` did not return within a 30 s client timeout while the phone
  reported 800 MB free and 2.1 GB of swap, so two threads is a proven limit, not
  a measured optimum.

### How a change is accepted

Interleave the arms within a round and rotate the order, one process per arm,
with one warm-up request discarded. Report N next to every percentile, and the
service number as `endpoint -> final` pooled over requests rather than the
timing of one profile: a single-process profile attributes a global cache effect
to one operator, which is how the cache fold first looked like a per-operator
win. Require the transcript to be byte-identical, or the command fields
unchanged, before accepting a latency win; a different transcript is a different
change. Record a negative result with its mechanism and its revival condition.
Index fixtures as `<pack>/<id>`, because a bare id silently merges the tuning and
holdout packs.

## Remaining work

1. **Real cabin VAD cuts.** Every number here uses synthetic fixtures and
   appended virtual silence. Acceptance needs real endpoint cuts under road
   noise, music, and overlapping speech, with the VAD's own jitter in the
   arrival schedule. The synthetic pack is a regression set, not an accuracy
   estimate.
2. **Five and six streams.** H9 covers four. Past two pairs the rule spills back
   to the device (4+0); whether a third batch lane or a third CPU slot is better
   is unmeasured, and the third CPU slot was already shown to slow the three
   short commands while speeding the long recording.
3. **Staggered arrivals under the cost-ranked rule.** H9 measured aligned
   commits only. The `vadburst` and `stagger4` cells in BENCHMARK.md predate the
   placement fix, so they have to be replayed before the rule is quoted for the
   common case.
4. **Where the CPU lane's 135 ms per chunk goes.** The device lane is 61.6 ms
   against roughly 135 ms on the CPU engine with two threads. The ConvNeXt f32
   projections and the small-K attention matmuls are the candidates; this is the
   tail setter in 2+2 and the only place left with a 2x gap.
5. **Model variants.** A shorter chunk or a lower-bit encoder changes the
   dispatch floor itself. Compare either against Q8_0 on identical VAD cuts
   before changing a default. Also open on quality: English `Set` needs a
   constrained decoder, and `氛围灯`/`座已` may still be reachable with a larger
   head gain.


## Cabin acceptance cases

The service target caps a physical inference batch at four. Test B=1, B=2,
and B=4 with independent session start offsets. In addition to aligned starts,
`bench_service.py --scenarios stagger-start4` feeds lengths 20 s, 10 s, and
two short commands at offsets 0, 10,000, 17,100, and 17,400 ms. The clips
overlap and end within about 300 ms of each other. Include mixed endpoints as
a separate case. Preserve per-stream ordering and compare endpoint-to-final
and completed-request throughput at the same arrival schedule.

The command classes are windows, climate, seats, and cabin lights. Judge each
recognized command against the intended action, target or position, value and
unit, and polarity. For multi-action requests require every action and count
any extra action as a semantic error. Include front/rear and left/right
confusion pairs, one/two/three levels, off versus leave on, and non-command
speech. Report Chinese CER and English WER alongside those checks.

`tools/vehicle_commands.tsv` and the evaluation generator provide a synthetic
starting set. `tools/eval_vehicle.py` implements those checks against
`tools/vehicle_semantics.tsv` and reports the counters separately, because CER
ranks variants while the command counters decide acceptance: on the tuning pack
the same rows read 8.43% CER and 68.8% command accuracy, and a phrase-bias
change can repair a noun without repairing the verb that carries the
direction. Keep the parser honest by re-parsing every reference sentence and
failing when its fields differ from the annotation; otherwise a parser bug
looks like a recognition error.

Keep an independent set of real cabin VAD cuts for accuracy
decisions. Before adding a bias list, measure which names, actions, or values
are missed. For recurring entity misses, A/B a bounded, session-scoped phrase
bias at token selection and verify it with negatives and confusable targets.
Test the same bias with and without HTP joiner execution. Avoid rewriting
recognized actions, positions, numbers, or negation at the text layer.
