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

## Next experiments

1. **Measure where the remaining tail goes.** In [`run_chunks`](xasr.cpp#L746) time feature and mask uploads, scheduled compute and completion, and encoder projection readback. In [`stream_accept`](xasr.cpp#L1018) time fbank, predictor/joiner, and text finalization. Tag each chunk with its phase (audio arrival or commit), sequence position, and number of DSP submissions. Collect a separate per-op profile using the [benchmark command](BENCHMARK.md#optional-htp-operator-profile). This distinguishes DSP kernel time from host submission and waiting. Compare CPU, HTP waiting, and HTP polling under both idle and representative CPU load.

2. **Reduce the work triggered by commit.** At 480 ms, the encoder consumes a 61-frame window and advances by 48 frames. The 1,200 ms of virtual silence is appended immediately but still triggers real inference. The readiness test is `pos + 61 < n_frames`, with `pos` advancing by 48 after each chunk. Using the fixture sample counts and the frame counts before and after padding yields two commit-time chunks on both short clips and the 20 s clip, and three on the exact 10 s clip. Count those chunks explicitly and sweep shorter padding only after testing clips that end at different offsets within the 480 ms window. Include clipped endpoints, final digits, addresses, and Chinese/English code-switching. Require stable final token sequences on reference clips and measure CER/WER and command success on real VAD segments before adopting a shorter flush.

3. **Optimize the dominant stage identified by profiling.** If host overhead dominates, reuse mask buffers and stop rewriting the masks once the left context is fully populated. If HTP kernels dominate, start with the ConvNeXt depthwise convolution and its two pointwise projections. The short-clip profile measured about 11.4 ms of DSP time per chunk across those three operators. Check layout conversions between them and A/B the whole block before pursuing a fused kernel. If the CPU joiner dominates, benchmark a vectorized matrix-vector path against the current 512-by-5,000 projection in [`greedy_run`](xasr.cpp#L824). Keep the greedy recurrence intact because the winning token from one encoder frame can change the predictor input for the next frame. Evaluate joiner offload only after measuring the dispatch cost at that granularity.

4. **Prove batching on service traffic.** Separate shared weights and graph metadata from per-stream encoder caches and predictor state. Start with batch size 1 and 2, then measure 4. Use ready chunks to fill a batch and prefer committed streams when their remaining chunks are ready. Compare no intentional batch wait with a bounded batching window. Preserve stream order, ensure ongoing streams get service, and test cancel, reset, and reuse for state leakage. Replay staggered arrivals, four simultaneous endpoints, and three endpoints while another stream keeps recording. Record queue wait, commit-to-final, endpoint-to-final P50/P95/P99, throughput, CPU time, and memory. Report sample count and run long enough to include thermal steady state. Compare with the current single-stream HTP and the shared-weight CPU ONNX scheduler before enabling batching by default.

5. **Revisit model variants after the latency breakdown.** Keep Q8_0 as the reference. A shorter chunk is a separate model geometry experiment with different left context, invocation frequency, and endpoint behavior. A lower-precision variant is a separate accuracy experiment. Compare both with Q8_0 on identical VAD cuts under road noise, music, and overlapping speech. Use Chinese CER, English WER, entity accuracy, and command success to decide if the measured tail reduction is acceptable.
