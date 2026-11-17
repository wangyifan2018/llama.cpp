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

See `docs/backend/snapdragon/paraformer-htp-adaptation-design.md` for the
stage-split, length-bucket, overlap/stitch and bilingual plan.
