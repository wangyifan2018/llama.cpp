# Alignment: sherpa-onnx streaming config for the X-ASR 480 ms Zipformer2 transducer

Scope: pin down the exact streaming knobs that sherpa-onnx applies to
`sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05`,
compare them with the X-ASR reference scripts and with
`examples/x-asr` (xasr.cpp + kaldi-fbank.cpp), and state which differences are
confirmed, which are measured, and which are unconfirmed.

Local model dir (read only):
`/Users/wangyifan/Library/Caches/qwen3-asr-htp/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05`

Everything below was checked on 2026-11-17 on the local host (macOS, 8 cores) with:
python 3.9.18, onnx 1.15.0, onnxruntime 1.18.0, kaldi_native_fbank + soundfile (pip),
sherpa-onnx 2.0.1 (pip wheel), and a Release build of `llama-x-asr` from this fork
built into /tmp/xasr-build (no repository file was modified).

What was actually executed (evidence, not inference):

1. ONNX metadata read directly from encoder.int8.onnx / decoder.onnx / joiner.int8.onnx.
2. `test_onnx.py` from the model dir, on all four fixture WAVs, `--use-int8 1`.
3. sherpa-onnx 2.0.1 python API, `model_type="zipformer2"`, `num_threads=1`,
   `decoding_method="greedy_search"`, endpoint detection off, tail silence 0/300/500/1200 ms.
4. Our `llama-x-asr` on the same four WAVs with the local Q8_0 GGUF.
5. Our `xasr_fbank_append` vs kaldi-native-fbank on 3.wav (883 frames compared).

## 0. Fixture geometry (measured)

Feature frames are counted with snip_edges=false, so frames = (samples + 80) / 160.

| Fixture | Duration | Feature frames | sherpa chunks, 0 ms tail | sherpa chunks, 1200 ms tail | Frames never encoded at 0 ms tail |
|---|---:|---:|---:|---:|---:|
| 0.wav | 10.053 s | 1005 | 20 | 23 | 32 (320 ms) |
| 1.wav | 5.100 s | 510 | 10 | 12 | 17 (170 ms) |
| 2.wav | 4.690 s | 469 | 9 | 11 | 24 (240 ms) |
| 3.wav | 8.830 s | 883 | 18 | 20 | 6 (60 ms) |

Chunk count = floor((frames - 61) / 48) + 1. Each chunk consumes 61 frames and
advances 48, so the last 1..48 frames of the signal are only encoded if extra
audio exists. sherpa-onnx never pads; see row "end of utterance" below.

## 1. Side-by-side table

"X-ASR reference" means the scripts shipped with the X-ASR repo that load these
ONNX files through sherpa-onnx, plus the released `test_onnx.py`. Where the two
disagree, both values are given.

| Knob | sherpa-onnx | X-ASR reference | ours (xasr.cpp / kaldi-fbank.cpp) | Match |
|---|---|---|---|---|
| Feature type | Kaldi fbank, knf::OnlineFbank | same (knf) | own Kaldi-compatible fbank | yes, numeric (max abs diff 6.0e-4) |
| Window / hop | 25 ms (400) / 10 ms (160) | 25 ms / 10 ms | 25 ms (win=400) / 10 ms (hop=160) | yes |
| Bins | 80 | 80 | 80 (n_mels=80) | yes |
| FFT size | 512 (round_to_power_of_two) | 512 | 512 (n_fft=512) | yes |
| Dither | 0.0 | test_onnx.py sets 0 (knf default is 1.0, overridden) | 0 | yes |
| snip_edges | false | false | false, emulated with a mirrored left edge of 120 samples | yes |
| DC removal | true (subtract frame mean before pre-emphasis) | true (knf default) | yes (mean of the 400 raw samples) | yes |
| Pre-emphasis | 0.97, first sample also scaled by 0.03 | 0.97 | 0.97, re[0] -= 0.97*re[0] | yes |
| Window function | "povey" = (0.5 - 0.5 cos)^0.85 | "povey" | hann^0.85 | yes |
| Mel range | 20 Hz .. 7600 Hz (low_freq=20, high_freq=-400) | test_onnx.py: 20 .. 8000 (knf default high_freq=0). sherpa path: 20 .. 7600 | 20 .. 7600 | matches sherpa; NOT test_onnx.py (measured text delta, see D2) |
| Log floor | FLT_EPSILON | FLT_EPSILON | epsilon | yes |
| Chunk consume (T) | 61 frames (610 ms) | 61 | 61 (T=61) | yes |
| Chunk shift | 48 frames (480 ms) | 48 | 48 (decode_chunk_len=48) | yes |
| Encoder frames per chunk | 12 (40 ms each, subsampling 4) | 12 | 12 | yes, measured |
| Right context / lookahead | 13 frames (130 ms) = T - shift | 13 | 13 | yes |
| Left context | 256 (per stack 256,128,64,32,64,128) | 256 | 256 | yes |
| End of utterance | no padding; drops the last 1..48 frames | test_onnx.py pads 100 frames (1 s) of zeros; deployment demo appends 500 ms of silence when its energy tail probe fires, otherwise 0 | 1200 ms of zeros appended at commit | NO (see D1) |
| Endpoint / VAD | 3 rules: 2.4 s silence (even with no speech), 1.2 s silence after speech, 20 s utterance length. Disabled by default in the python API and in the X-ASR demo | disabled (enable_endpoint_detection=False); external energy tail probe (500 ms silence, 500 ms min silence) | none inside the runtime; caller calls xasr_commit at its VAD endpoint | partial: both end up "off", but ours has no rule equivalents |
| Decoding method | greedy_search (1 best path) | greedy_search | greedy, 1 best | yes |
| Max symbols per frame | 1 (single argmax per encoder frame, no inner loop) | 1 | 1 | yes |
| blank id | 0, hardcoded | 0 | 0 (blank_id=0) | yes |
| unk handling | id from tokens.txt, never emitted, counted as blank | same | same (unk_id from GGUF) | yes |
| Text join | replace U+2581 with space, drop <unk>, RemoveSpaceBetweenCjk on | same, then the demo applies its own regex spacing pass plus optional lower/capitalize | same join and CJK rule, then strips leading spaces | partial (see D3, D4) |
| ITN / homophone replacer | no-op unless configured | not configured | absent | yes |
| Threads | num_threads (ORT intra-op). python default 2, X-ASR scripts 1, C API example 1 | 1 | n_threads default 2 (GGML CPU backend), host greedy is single threaded | no (see D5) |
| Model precision | int8 ONNX for encoder and joiner, fp32 decoder | int8 | Q8_0 GGUF | no (see D6) |

## 2. sherpa-onnx knob names, defaults, and sources

Feature extraction (all fields are in FeatureExtractorConfig unless noted):

| Knob | Default | Source |
|---|---|---|
| `--feat-dim` / `feature_dim` | 80 | [features.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.h) |
| `--sample-rate` / `sampling_rate` | 16000 | [features.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.h) |
| `--low-freq` / `low_freq` | 20.0 | [features.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.h) |
| `--high-freq` / `high_freq` | -400.0 (7600 Hz at 16 kHz) | [features.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.h) |
| `--dither` / `dither` | 0.0 | [features.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.h) |
| `snip_edges` | false | [features.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.h) |
| `frame_shift_ms` | 10.0 | [features.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.h) |
| `frame_length_ms` | 25.0 | [features.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.h) |
| `remove_dc_offset` | true | [features.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.h) |
| `preemph_coeff` | 0.97 | [features.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.h) |
| `window_type` | "povey" | [features.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.h) |
| `round_to_power_of_two` | true | [features.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.h) |
| `normalize_samples` | true | [features.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.h) |

How those fields are pushed into kaldi-native-fbank (InitFbank: dither, snip_edges,
samp_freq, shift, length, dc offset, pre-emphasis, window type, power-of-two,
num_bins, high_freq, low_freq): [features.cc](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.cc).

Chunk geometry (no user knob, read from the encoder ONNX metadata):

| Knob | Value for this model | Source |
|---|---|---|
| `T` -> `ChunkSize()` (frames consumed per chunk) | 61 | [online-zipformer2-transducer-model.cc](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-zipformer2-transducer-model.cc), [.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-zipformer2-transducer-model.h) |
| `decode_chunk_len` -> `ChunkShift()` (frames advanced per chunk) | 48 | same files |
| `left_context_len` (per stack) | 256,128,64,32,64,128 | same files |
| Readiness rule | `num_processed + ChunkSize() < NumFramesReady()` | [online-recognizer-transducer-impl.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer-transducer-impl.h) |
| No feature padding at EOF | `GetFrames` aborts if fewer than n frames are ready, and the loop stops first, so 1..48 final frames stay unencoded | [features.cc](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.cc), [online-recognizer-transducer-impl.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer-transducer-impl.h) |

Decoding and endpointing:

| Knob | Default | Source |
|---|---|---|
| `decoding_method` (C++ field, python kwarg `decoding_method`) | "greedy_search" | [online-recognizer.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer.h) |
| `max_active_paths` | 4, beam search only | [online-recognizer.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer.h) |
| `blank_penalty` | 0.0 | [online-recognizer.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer.h) |
| `temperature_scale` | 2.0, rescales saved log probs only, argmax unaffected | [online-recognizer.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer.h), [online-transducer-greedy-search-decoder.cc](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-transducer-greedy-search-decoder.cc) |
| one symbol per frame, unk treated as blank, blank id 0 | hardcoded | [online-transducer-greedy-search-decoder.cc](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-transducer-greedy-search-decoder.cc) |
| `enable_endpoint` / `enable_endpoint_detection` | C++ struct true, python `from_transducer` False, X-ASR demo 0 | [online-recognizer.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer.h), [online_recognizer.py](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/sherpa_onnx/online_recognizer.py) |
| rule1: no speech needed, 2.4 s trailing silence | `rule1{false, 2.4, 0}` | [endpoint.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/endpoint.h) |
| rule2: after speech, 1.2 s trailing silence | `rule2{true, 1.2, 0}` | [endpoint.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/endpoint.h) |
| rule3: 20 s utterance length | `rule3{false, 0, 20}` | [endpoint.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/endpoint.h) |
| trailing silence measured as `num_trailing_blanks * 4` frames at 10 ms | 40 ms per encoder frame | [online-recognizer-transducer-impl.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer-transducer-impl.h) |
| `num_threads` | python 2, X-ASR scripts 1 | [online_recognizer.py](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/sherpa_onnx/online_recognizer.py) |

X-ASR reference values (read from the repo):

- Streaming wrapper defaults: `num_threads=1`, `feature_dim=80`, `sample_rate=16000`,
  `decoding_method="greedy_search"`, `model_type="zipformer2"`,
  `enable_endpoint_detection=False`:
  [sherpa_streaming_infer.py](https://github.com/Gilgamesh-J/X-ASR/blob/main/X-ASR-zh-en/deployment/infer_and_client/sherpa_streaming_infer.py)
- Demo server defaults: `--num-threads 1`, `--enable-endpoint-detection 0`,
  `--enable-energy-tail-probe 0`, `--tail-probe-ms 500`, `--min-silence-ms 500`;
  it appends silence and decodes when the probe fires, then calls input_finished:
  [sherpa_streaming_server.py](https://github.com/Gilgamesh-J/X-ASR/blob/main/X-ASR-zh-en/deployment/infer_and_client/sherpa_streaming_server.py)
- The released `test_onnx.py` in the model dir sets only dither=0, snip_edges=false,
  window_type="povey", samp_freq, num_bins=80, then pads 100 feature frames.

## 3. Confirmed behavioural differences

### D1. End-of-utterance padding: none vs 1200 ms (confirmed, measured)

sherpa-onnx runs the encoder only on whole 61-frame chunks and never synthesises
features at EOF. Our runtime appends 1200 ms of zeros at commit.

- Transcript content: measured, sherpa 0 ms tail vs 1200 ms tail on the fixtures:
  0.wav loses 3 tokens (token 576,296,124 = "xing qi san"), 1.wav loses 5 tokens
  (53,30,119,269,36 = "shen me yi si a"), 2.wav loses 1 token (15), 3.wav loses 0.
  Token counts 24/15/22/31 vs 27/20/23/31. So an unaligned comparison can show a
  whole trailing phrase missing, and that phrase is often the entity or the
  question word.
- Tail latency: our 1200 ms pad forces about 2.5 extra chunks. Measured chunk
  counts 20->23, 10->12, 9->11, 18->20 (+2 to +3 chunks). At the measured
  per-chunk cost on this host (wall_ms / chunks is 130-210 ms) that is roughly
  0.3-0.6 s added to flush time. Our measured flush_ms on the fixtures was
  787 / 220 / 517 / 251 ms. Most of that number is the padding, not real speech.
- Direction: sherpa is "cheaper tail, truncated text"; ours is "complete text,
  longer tail". For a latency benchmark this difference must be normalised
  (e.g. compare only to the last real token, or use the same padding on both).

### D2. Mel high edge: 7600 Hz vs 8000 Hz in the released test_onnx.py (confirmed, measured)

sherpa-onnx uses high_freq=-400 (7600 Hz). The released test_onnx.py leaves
knf's default high_freq=0 (8000 Hz). Same audio, same weights, different mel banks.

- Measured: 2.wav first word flips between token 898,1635 (U+9891 U+7E41, "pin fan")
  at 8000 Hz and token 272,1193 (U+5E73 U+51E1, "ping fan") at 7600 Hz.
  1.wav gains an extra token 8 (space + comma) at 7600 Hz near token position 13.
  0.wav and 3.wav are identical.
- Metric distorted: transcript content, about 1 word in 4 fixtures here. Our
  implementation matches sherpa-onnx, not test_onnx.py, so a parity check against
  test_onnx.py output would show a phantom mismatch.

### D3. Leading space (confirmed by code + run)

sherpa-onnx C++ keeps the leading space in `r.text` when the first symbol starts
with U+2581. Our `detok` strips leading spaces. The python wrapper in 2.0.1 also
strips (`get_result` returns `text.strip()`), which is why the measured python
transcripts match ours on this point.
Metric: exact-string equality only, 1 character. It does not change words.

### D4. Extra post-processing in the X-ASR demo (confirmed by code)

`sherpa_streaming_infer.py` applies `_normalize_cjk_spacing` with its own CJK
ranges (U+3400-U+4DBF, U+4E00-U+9FFF, U+F900-U+FAFF) and punctuation list, plus
optional lower/capitalize. sherpa-onnx uses IsCjkWithoutWordSpace/IsPunct from
text-utils.cc. The two agree on common text but are not the same function, so
text produced by the demo is not automatically the sherpa-onnx canonical text.
Metric: transcript content (spacing/case), entity-string comparison.

### D5. Thread count default (confirmed defaults, magnitude not measured)

sherpa-onnx python defaults to `num_threads=2`, X-ASR scripts pass 1, our
`xasr_context_params.n_threads` defaults to 2 (GGML CPU backend only; the host
greedy loop is single threaded). Metric: RTF only, no effect on text. Magnitude on
this model/host: unconfirmed, not measured.

### D6. Weight precision: Q8_0 GGUF vs int8 ONNX (confirmed, measured)

Both are quantised, but with different tools and different granularity.
Measured ours vs the sherpa-onnx reference on the same four fixtures:

| Fixture | sherpa-onnx reference | ours | Char edit distance |
|---|---|---|---|
| 0.wav | 27 tokens | same length, "M" (U+004D) becomes "m" (U+006D) | 1/53 = 1.9% |
| 1.wav | 20 tokens | identical | 0% |
| 2.wav | 23 tokens | last word U+9891 U+7E41 becomes U+5E73 U+51E1 | 2/30 = 6.7% |
| 3.wav | 31 tokens | U+65F6 U+6001 becomes U+98DF U+6750, and U+65F6 becomes U+5F0F | 3/38 = 7.9% |

Metric distorted: transcript content and entity accuracy. This is the only
difference that survives when the front end and the padding policy are aligned,
so it is the right place to look for the remaining parity gap.

### D7. Endpointing behaviour (confirmed by code)

sherpa-onnx can end a segment itself (rule2: 1.2 s of trailing blank at 40 ms per
encoder frame, i.e. 30 blank encoder frames after speech) and resets the stream.
Our runtime has no internal endpointing and expects the caller to commit.
If llama.cpp wants sherpa-compatible segmentation, rule2 with 1.2 s is the
closest equivalent. Metric: segment boundaries, entity accuracy across segments,
and per-segment latency. Magnitude: unconfirmed, depends on the caller's VAD.

### D8. Not a difference (checked, deliberately)

Readiness test (`pos + T < n_rows` vs `num_processed + ChunkSize() < NumFramesReady()`),
advance of 48, blank/unk suppression, single argmax per encoder frame, initial
hypothesis [-1]*(ctx-1) + [blank], and per-frame decoder refresh on emission all
match. Our incremental fbank also matches knf numerically: 883 frames on 3.wav,
max abs diff 5.999e-4, mean 4.12e-6, p99 4.86e-5 (float FFT rounding only).

## 4. Reproduce the sherpa-onnx reference transcript for the four fixture WAVs

Transcript parity only. These commands run ONNX Runtime on the host CPU; they say
nothing about device latency, HTP placement, or tail latency of a shipped runtime.

### 4a. sherpa-onnx python package (canonical runtime path)

```sh
python3 -m venv /tmp/sherpa-ref
. /tmp/sherpa-ref/bin/activate
pip install "sherpa-onnx==2.0.1" soundfile

MODEL="$HOME/Library/Caches/qwen3-asr-htp/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05"

python3 - "$MODEL" <<'PY'
import sys
import numpy as np
import soundfile as sf
import sherpa_onnx

D = sys.argv[1]
rec = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=D + "/tokens.txt",
    encoder=D + "/encoder.int8.onnx",
    decoder=D + "/decoder.onnx",
    joiner=D + "/joiner.int8.onnx",
    num_threads=1,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
    provider="cpu",
    model_type="zipformer2",
    enable_endpoint_detection=False,
)

for w in ("0", "1", "2", "3"):
    d, sr = sf.read(f"{D}/test_wavs/{w}.wav", always_2d=True, dtype="float32")
    s = rec.create_stream()
    s.accept_waveform(16000, np.ascontiguousarray(d[:, 0]))
    s.accept_waveform(16000, np.zeros(19200, dtype=np.float32))  # 1200 ms of silence
    s.input_finished()
    while rec.is_ready(s):
        rec.decode_stream(s)
    print(w, rec.get_result(s))
PY
```

Expected result (token counts 27 / 20 / 23 / 31, token ids from tokens.txt):

```
0 869,60,17,4765,4973,4198,4663,4065,8,4406,4032,674,836,120,8,4019,4208,4231,4020,4054,4171,4229,4169,17,576,296,124
1 22,17,219,20,129,8,219,120,129,237,504,352,4298,8,4298,53,30,119,269,36
2 25,17,272,1193,15,36,8,21,257,434,278,61,31,8,4421,4043,4977,4209,4284,4050,898,1635,15
3 219,20,427,17,23,53,30,49,547,11,259,19,4836,17,20,842,72,319,49,8,58,8,16,26,92,168,49,547,478,39,43
```

Drop the `accept_waveform` of zeros to see the truncation described in D1.

### 4b. Released test_onnx.py (model dir, no sherpa-onnx needed)

```sh
pip install kaldi_native_fbank onnxruntime soundfile
cd "$HOME/Library/Caches/qwen3-asr-htp/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05"
for i in 0 1 2 3; do python3 test_onnx.py --use-int8 1 --wav test_wavs/$i.wav; done
```

Caveats, all confirmed here: it prints raw symbols with U+2581 still in place and
no CJK space removal; it uses knf defaults for the mel banks, i.e. 8000 Hz instead
of sherpa-onnx's 7600 Hz (see D2); it pads 100 feature frames (1 s).

## 5. Where the model metadata pins the geometry

Read from the ONNX files in the model dir (encoder.int8.onnx metadata):

```
model_type          zipformer2
decode_chunk_len    48
T                   61
left_context_len    256,128,64,32,64,128
num_encoder_layers  2,2,4,5,4,2
encoder_dims        192,256,512,768,512,256
cnn_module_kernels  31,31,15,15,15,31
num_heads           4,4,4,8,4,4
query_head_dims     32,32,32,32,32,32
value_head_dims     12,12,12,12,12,12
```

decoder.onnx: `context_size=2`, `vocab_size=5000`. joiner.onnx: `joiner_dim=512`.
Encoder input 0 is a fixed `(N, 61, 80)` float tensor, so 80 bins and 61 frames
are structural, not runtime choices.

The X-ASR export script derives these numbers and shows why they cannot float:

- `decode_chunk_len = chunk_size * 2` (chunk_size is in 20 ms frames, so 24 -> 48).
- `pad_length = 7 + 2 * 3 = 13` and `T = decode_chunk_len + pad_length` (48 + 13 = 61).
  The 7 is the encoder_embed's own padding; the 2*3 is the ConvNeXt right padding
  after subsampling ((7 - 1) // 2 = 3 frames on each side).
- `left_context_len = left_context_frames // ds` per stack, 256 -> 256,128,64,32,64,128.
Source: [export-onnx-streaming.py](https://github.com/Gilgamesh-J/X-ASR/blob/main/X-ASR-zh-en/zipformer/export-onnx-streaming.py).

Consequences for a llama.cpp fork:

1. Feeding anything other than 61 frames and advancing anything other than 48 frames
   is not a runtime tuning knob; it invalidates the exported graph (T is baked into
   the export and into the encoder input shape).
2. Changing the chunk to 160/960/1920 ms means loading a different encoder export.
   The release ships four separate models (160/480/960/1920 ms):
   [X-ASR model card](https://huggingface.co/GilgameshWind/X-ASR-zh-en) (seen via
   search result; the page itself could not be fetched from this host).
3. Our GGUF encodes the same table: `xasr.chunk_ms = 160,480,960,1920` and
   `xasr.left_context_frames = 96,256,256,256`, and xasr.cpp refuses to load if
   either table or if `T`, `decode_chunk_len`, dims, or vocab differ
   (examples/x-asr/xasr.cpp, validation block at lines 239-240).
4. Left context is 256 frames at 20 ms per internal frame = 5.12 s of history at
   480 ms. That is a model property; it cannot be shortened without retraining.

## 6. Unconfirmed

- The exact sherpa-onnx release tag or asset that ships this model name. The
  directory name follows the release naming, but GitHub API was rate limited and
  huggingface.co was not reachable from this host during the check. Sources cited
  above are from sherpa-onnx master, and the measured transcripts come from the
  pip wheel 2.0.1; the two were consistent on every point I tested.
- Whether the 480 ms checkpoint was trained with high_freq=-400. The ONNX metadata
  does not record fbank options. The measured behaviour (7600 Hz matches the
  deployed sherpa-onnx transcripts, 8000 Hz does not) is consistent with -400 but
  does not by itself prove the training configuration.
- The exact magnitude of the thread-count difference (D5) on any target device.
- The claim, from a third-party release note, that sherpa-onnx's own streaming
  examples feed 0.66 s of trailing silence before input_finished. I did not open
  that example file in this session.
- Any behaviour of the Android/HTP build. No adb and no device were used, per the
  task constraints.
