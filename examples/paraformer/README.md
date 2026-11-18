# Paraformer (FunASR) example

CPU-first baseline for adapting FunASR Paraformer-large to this llama.cpp fork.
The runtime code is derived from CrispASR's `src/paraformer.cpp` and its
`core/{sanm,kaldi_fbank,lfr,parallel_for}` helpers (FunASR Model License;
commercial use requires attribution). The GGUF loader and device shims here are
minimal local replacements so the example builds against this repository.

## Quick start

`reproduce.sh` wraps weights, build, fixtures and the accuracy gate. It is the
fastest way to check that a checkout still does what this file says it does, and
it is the answer to "can I reproduce the numbers" - see
[NOTES-REPRODUCE.md](NOTES-REPRODUCE.md) for what transfers between machines and
what does not.

```sh
cd examples/paraformer
./reproduce.sh check      # tools, weights, fixtures, binary: reports, writes nothing
./reproduce.sh all        # fetch -> build -> render fixtures -> assert the 4 numbers
```

The four gates are 71.9% cabin command, 100.0% mixed command, 0.60% Mandarin CER
and 6.87% English WER, and the script exits non-zero if any of them regress.

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

### Hotword profile: gains, prefix conditions and span rewrites (off by default)

`--hotwords FILE` applies a fixed logit gain in the NAR decoder's argmax. Three
line forms (`#` comments OK, gain in [0,5] and margin in [0,5]; tokens must exist
in the GGUF vocab):

```text
<gain> <token>                    global gain at every CIF position
<gain> <prefix>><token>           conditional: only when previous token == prefix
R <margin> <w1+w2+...> => <r1+r2+...>
                                  span rewrite, applied after argmax
```

Picking between them is a question about the **shape** of the error, and the
three cases are distinct:

| Error shape | Use | Example |
|---|---|---|
| Same token count, the right word is one token, a correct prefix precedes it | `<prefix>><token>` | `se@@ ed`→`se@@ at` ("seed"→"seat"), `t@@ al`→`t@@ well` ("footal"→"footwell") |
| Same token count, but the wrong path won from the start and no prefix is correct | `R` | `we+are+right`→`re@@+ar+right` ("we are right"→"rear right") |
| Different token count | `R` only | `li@@ kes`(2)→`li@@ gh@@ ts`(3), `f@@ oo@@ t@@ w@@ all`(5)→`f@@ oo@@ t@@ well`(4) |

A logit gain cannot change the token count, and a global gain is the wrong tool
whenever the same target token needs opposite signs in different contexts: bare
`two` needs `1.5` after `l` (`level two`) and must be suppressed after `to`
(`temperature to 二十一度`), so the profile uses `1.5 l>two`. `R` is gated on the
acoustic score — it fires only when the replacement's per-position average logit
is within `<margin>` of the observed span, so a margin below the measured deficit
silently does nothing (measured 2.39 for `rear right`: `1.5` is rejected, `2.6`
fires).

`hotwords_vehicle.txt` ships the recommended config. Without the flag the
baseline is bit-identical.

```sh
llama-paraformer MODEL.gguf WAV --hotwords hotwords_vehicle.txt [--device HTP0]
```

Diagnostics: `--dump-logits N` prints the top-5 logits (with gain applied) at the
first N CIF positions; `PF_WATCH_TOKENS=a,b` (env) appends the exact gap of named
tokens; `PF_DUMP_TOKENS=1` shows the raw token sequence; `PF_REPLACE_LOG=1`
reports each `R` match with its deficit; `PF_DUMP_VOCAB=<substring>` dumps vocab
entries. Measure the gap before choosing a gain — on this model `said→set` is
2.89–3.00, `too→two` 0.96, `se@@+ed→at` 3.26–3.46, while `in→and` is 4.16 and
also disturbs a neighbouring position, so it is excluded.

Measured with the shipped profile (clause-weighted command field, `zh` CER and
dictation `en` WER over the `tingting` fixture):

| Arm | 32-clip cabin | 15-clip mixed | zh CER / en WER |
|---|---:|---:|---:|
| no profile | 56.2% | 80.0% | 0.60% / 12.21% |
| shipped profile | **71.9%** | **100.0%** | 0.60% / **6.87%** |

Reproduce with `./reproduce.sh all`; the gate exits non-zero if any of those four
numbers regress, and the run directory keeps both arms' transcripts so a failure
can be diffed rather than guessed at. `./reproduce.sh check` reports what is
missing without writing anything. Design doc §13–15 and §30–32 cover how each
rule was derived, including two rules that were tested and rejected.

Those four numbers are measured on **Tingting-rendered fixtures**, and the
fixtures are not committed - only the text they were rendered from is. macOS
`say` is what produced them; on another platform `./reproduce.sh fixtures`
cannot run and the gate's absolute thresholds will fail by construction. Read
[NOTES-REPRODUCE.md](NOTES-REPRODUCE.md) before concluding that means a
regression: what transfers off-Mac is the profile's gain over the no-profile
arm, not the absolute rate.

These gains and gaps belong to the **zh checkpoint** (`hotwords_vehicle.txt`
uses `set` and `se@@`, neither of which exists in the en vocab). The profile is
not portable: loading it against the en checkpoint logs `hotword token not in
vocab` and applies nothing. Regenerate the gaps per checkpoint before tuning.

`hotwords_vehicle_replace.txt` is **deprecated**: it carries the global `1.5 two`
that the shipped profile replaces with `1.5 l>two`, and it scores 70% mixed
against 80% with no profile at all. It is kept only to replay an older
measurement.

## Weights

Every number in this file and in `PRODUCTION.md` belongs to these exact files.
The fingerprint is the contract, not the URL — any copy with a matching sha256
is equivalent, and a different conversion of the same source checkpoint shifts
the logit gaps that `hotwords_vehicle.txt` was tuned against, which makes a
correct profile look broken for no visible reason.

| File | Bytes | sha256 | Role |
|---|---:|---|---|
| `paraformer-zh-q8_0.gguf` | 237454336 | `e73232ecf7e484907fd58f6a481fa27c1ceff454a1bc441d4a032a3a63e9783f` | ships |
| `paraformer-en-q8_0.gguf` | 239295840 | `87c2845b1100a629bef15dfaafdbeb30be9ac5b3b5a07077e413979533d26638` | comparison arm |
| `paraformer-en-f32.gguf` | 444785792 | `a41d43bc2a34dbdd66e7e741baa466577f9137a16f93f454b35658b0db861c42` | source for `requant.cpp` |
| `paraformer-zh-q4_k.gguf` | 128687104 | `c3dc70409857d7450895d8ae586f1bef23755a7635cc79cb1ff2ffb08720260e` | A/B arm, slower on v73 |

```sh
python3 tools/fetch_models.py --out ~/models/paraformer          # fetch + verify
python3 tools/fetch_models.py --out ~/models/paraformer --verify-only
```

`tools/model_manifest.txt` is the authoritative list; the fetch script deletes
anything that fails verification rather than leaving a usable-looking file
behind. The source checkpoint is [FunASR Paraformer on
ModelScope](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch);
the GGUF conversion is the one recorded in `tools/model_manifest.txt`, and this
repository does not contain the HF→GGUF converter (`convert_hf_to_gguf.py` has
no `paraformer` architecture). Treat the sha256 as the requirement.

## Status (2026-11-18, host CPU and SM6650, Q8_0)

- 32-clip `tingting` cabin fixture, with the shipped hotword profile: Chinese
  CER 0.60%, command field **71.9%**, en WER **6.87%**. The English column of
  that same fixture is not quotable on its own - Tingting has no `en_US`
  variant, so its English clips are Mandarin-accented speech and measure the
  voice. English on native `en_US` voices is 39.50% (zh checkpoint) and 24.14%
  (en checkpoint); see the checkpoint section below.
- Raw command field without the profile is 56.2%; the gap to 71.9% is both the
  hotword profile and the lack of punctuation (Paraformer emits none, and the
  semantic evaluator splits clauses on `，`/`,`). The punctuation-only arm is
  68.8%, so the profile is worth about 3 points on top of punctuation.
- 15-clip mixed Mandarin/English fixture (`tools/cabin_mixed.tsv`): command
  field **100%**. This is the fixture that decides the checkpoint, and it is the
  one the old `hotwords_vehicle_replace.txt` loses 30 points on.
- x-asr reference on the same 32-clip fixture (on-arm): command 75.0%, CER 7.23%.
- Device SM6650, 32 clips, warm-run excluded: CPU p50/p95 357/496 ms, HTP
  290/380 ms; transcripts are identical between CPU and HTP. A single CLI
  invocation measured in isolation also carries process start and the first
  scheduler allocation, so its wall time is not a steady-state latency.
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
- Six English clips still fail with the shipped profile, and all six are word
  forms the vocabulary lacks (`heater`, `massage`, `vents`, `lights` as a whole
  word) or multi-token clumps (`driversee peater`). Bias cannot create a token
  that is not in the 8404-entry vocabulary, and the `R` margin cannot bridge a
  deficit that large. Closing these needs in-domain fine-tuning or a
  cabin-domain proper-noun post-processor; see `PRODUCTION.md`.

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

## API

See [PRODUCTION.md](PRODUCTION.md) for hardening changes, regression replays, and open release checks.

Callers should read [docs/backend/snapdragon/api-reference.md](../../docs/backend/snapdragon/api-reference.md):
endpoints, request fields, response formats, headers (`X-Queue-Ms`, `X-Infer-Ms`,
`X-Language`, `X-Vad-Pieces`, `X-Speaker-Score`), error codes, the optional
speaker gate and server VAD, and integration notes.

## OpenAI-compatible API

`llama-paraformer-api` serves the engine over HTTP with the OpenAI audio shape.
The front end already commits one segment at the VAD endpoint, so a whole
utterance per request maps directly onto the engine.

### Authentication

Off by default, so nothing that already works stops working. Set a token and
every endpoint that accepts audio — including the `/v1/realtime` upgrade —
requires `Authorization: Bearer <token>`:

```sh
export PF_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
build-x-asr/bin/llama-paraformer-api MODEL.gguf --port 8080 --api-key "$PF_API_KEY"
```

`PF_API_KEY` is preferred over `--api-key` because an argument is readable in
`/proc/<pid>/cmdline` by every local process. The flag wins if both are set.

`/health`, `/v1/models` and `/metrics` stay open: a container healthcheck has no
way to hold a credential, and refusing it turns a running service into a
permanently unhealthy one. They expose engine counts and queue depth, not
audio — if that is still too much, put the service behind a proxy that
authenticates everything.

A rejected request gets 401 with the same envelope as every other error, so a
client needs one parser:

```json
{"error":{"code":"invalid_api_key","message":"missing or invalid bearer token","type":"authentication_error"}}
```

Token comparison is constant-time; a plain `==` leaks how much of the prefix
matched through timing, which is enough to recover a token byte by byte over a
network. `tools/check_auth.py` asserts the refusals, the acceptance, the open
probes and the WebSocket path against a running server:

```sh
python3 tools/check_auth.py --port 8080 --wav clip.wav --key "$PF_API_KEY"
python3 tools/check_auth.py --port 8080 --wav clip.wav --no-auth   # open server
```

### Shutdown

`SIGTERM` and `SIGINT` mean "finish what you already accepted, then exit", which
is what a service manager expects from a rolling restart. On the signal the
server stops accepting new work only after the accepted requests are answered:

```
pf-api: shutdown requested; draining before closing the listener
pf-api: drain start, accepted=3 inflight=1, grace 5000 ms
pf-api: closing the listener
pf-api: shut down cleanly
```

`--shutdown-grace-ms` bounds the wait (5000 ms default); when it runs out the
listener closes anyway and the log says so instead of pretending the drain
succeeded. Three details are load-bearing and each was measured:

- The counter being drained is `accepted`, not `inflight`. `inflight` counts
  only tasks a worker has popped, so it reads zero for a request that is still
  queued or already has its answer and is being written back. Draining on it
  declares success with clients still waiting — the exact failure this exists
  to prevent.
- The listener is stopped *after* the drain, not before. `srv.stop()` tears down
  active connections, so calling it first ends the handlers the drain is waiting
  for: the guard drops to zero, the log says "clean", and only the already
  buffered bytes reach the client.
- A request that finishes just after the grace expires still gets a complete
  response, because closing the listener does not kill a running handler. That
  is why the timeout message says the outcome is "no longer guaranteed" rather
  than predicting a reset.

### Binding

The default `--bind` is `127.0.0.1`. A service that accepts unauthenticated
audio should not appear on every interface because of a default nobody chose;
putting it on the LAN is an explicit `--bind 0.0.0.0`, and the server prints a
warning at startup when the address is not loopback and no token is set.

```sh
build-x-asr/bin/llama-paraformer-api MODEL.gguf --device HTP0:0 --port 8080 \
    --engines 2 --threads 2 --queue 64 --queue-mib 8 --deadline-ms 30000 --model-id paraformer-zh-q8_0 \
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
inference. Overload returns 429, a queue timeout returns 503, and a final request
that reaches `--deadline-ms` returns 504. Input must be
16 kHz mono 16-bit PCM WAV; other formats are rejected rather than resampled.
Realtime streaming is the same server on `/v1/realtime` (WebSocket): send
`input_audio_buffer.append` with base64 PCM16 while the user speaks, then
`input_audio_buffer.commit` at the VAD endpoint. The server answers with
`conversation.item.input_audio_transcription.delta` (throttled partials, see
`--partial-ms`/`--partial-interval-ms`) and `.completed`. Commit tasks take priority
over queued previews. The API accepts up to four WebSocket sessions, with up to
four pending commits per session. Each acknowledged commit receives one terminal
completion or error with its item ID. See [the production log](PRODUCTION.md#http-and-websocket-admission-and-deadlines) for replay commands.

```sh
python3 tools/ws_client.py --url ws://127.0.0.1:8080/v1/realtime --wav a.wav --paced
```

See docs/backend/snapdragon/asr-server-api-design.md for the full surface,
including the Realtime event mapping and what a general ASR server needs.

`--prefer-idle-htp` prioritizes an idle HTP worker and lets CPU workers pull
from the same queue while HTP is busy. See the [SM6650 routing replay](PRODUCTION.md#backend-scheduling-on-sm6650)
for the four-request latency and memory measurements. For long-running Android
deployments, the [process recovery launcher](PRODUCTION.md#process-recovery-on-cpu)
shows how to rewarm on CPU after the HTP process exits.

### Speaker gate (optional)

The API can score a committed segment against the enrolled driver's voiceprint
with CAM++ through sherpa-onnx. The optional build runs one scoring task at a
time on a dedicated CPU thread.

```sh
# Android build (sherpa headers and libs mounted into the container)
docker run --rm -v "$PWD:/workspace" -v /tmp/sherpa:/sherpa:ro \
    --platform linux/amd64 -u 501:20 ghcr.io/snapdragon-toolchain/arm64-android:v0.8 \
    bash -c 'cmake -S /workspace -B /workspace/build-android-xasr -DXASR_SPEAKER=ON \
        -DXASR_SHERPA_INCLUDE_DIR=/sherpa/inc -DXASR_SHERPA_LIB_DIR=/sherpa/lib && \
        cmake --build /workspace/build-android-xasr --target llama-paraformer-api -j 6'

# on the phone: LD_LIBRARY_PATH needs lib/ and onnx/bin (sherpa + onnxruntime)
export LD_LIBRARY_PATH=$PWD/lib:$PWD/onnx/bin
bin/llama-paraformer-api MODEL.gguf --device HTP0:0 --port 8899 --engines 2 --bind 0.0.0.0 \
    --speaker-model onnx/model/spk_od.onnx --enroll-dir pf-enroll --speaker-threshold 0.55
# `adb forward tcp:8899 tcp:8899` reaches a loopback bind; use --bind 0.0.0.0
# only if the phone's network should also reach it, and set PF_API_KEY if so.
```

`--enroll-dir` holds 16 kHz WAVs for the enrolled speaker. Startup prints the
clip count and mean self-cosine. The scoring worker uses at most the first
1.5 seconds by default. The final transcript carries `assessed` if a result
is ready, `not_assessed` with a reason for an enabled but incomplete or skipped
score, or `disabled`. Both REST endpoints expose the state in their headers,
and the WS final event carries the same state. An assessed score and its
threshold decision are advisory to the caller. For device measurements and
the replay command, see [PRODUCTION.md](PRODUCTION.md).

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

## Voice fixtures

`tools/cabin_commands.tsv` holds the 32 cabin texts, and
`tools/gen_voice_fixtures.py` renders them in any macOS voice as 16 kHz mono
PCM16. It selects the language variant per row, because `say -v Eddy` resolves
to the English one and renders Mandarin as digital silence while exiting 0.

```sh
python3 tools/gen_voice_fixtures.py --voice Sandy --out /tmp/voice-sandy
python3 tools/score_voice.py --dir /tmp/voice-sandy --url http://127.0.0.1:8899
```

`score_voice.py` reports the same zh CER and en WER used in
[PRODUCTION.md](PRODUCTION.md). Which voices may be scored is fixed by
`tools/voice_languages.tsv`: the multilingual novelty voices cannot synthesise
Mandarin well enough to measure with - Whisper small, an independent acoustic
model, scores 46-69% zh CER on the same clips, worse than this engine - so
their Mandarin rows measure the TTS. Their English output is clean native
speech, and there Whisper scores 3.39% WER against 24.14% for this engine, so
English is a real gap rather than a fixture artefact. Use a Mandarin voice for
Mandarin fixtures: Meijia scores 0.00% zh CER. These are TTS clips: speaker
variety without channel variety.

**English needs the English checkpoint for American speakers; the product needs the
Mandarin one.** FunASR publishes a separate `paraformer-en` (vocab 10020, 50000 h
English) alongside `paraformer-zh` (vocab 8404, 60000 h Mandarin+English). Both
load in this engine unchanged - the vocabulary size comes from GGUF metadata.
On 8 native `en_US` voices the English checkpoint scores 24.14% WER against the
Mandarin checkpoint's 39.50% at matched Q8_0 (39.50% -> **6.87%** with the
shipped hotword profile; the other 32.6 points are hotwords, not precision). But
the 39.50% is a speaker effect, not a language effect: scored on the same
English sentences rendered through a Mandarin voice, `paraformer-zh` reads
12.21% and `paraformer-en` reads 4.58%. Accented English is inside the Mandarin
checkpoint's training distribution; General American is closer to outside it.

That decides the model. `paraformer-en` fed a mixed sentence returns pinyin -
`打开 front left window` becomes `dark high front left window` - so it cannot be
the product checkpoint when Mandarin is primary. `paraformer-zh` handles mixed
speech correctly (`打开 front left window`, `关闭 seat heater`, `空调调到 twenty
two degrees` all come back right). `tools/cabin_mixed.tsv` is the 15-sentence
mixed fixture that measures this; regenerate it with
`gen_voice_fixtures.py --manifest tools/cabin_mixed.tsv`.

The checkpoints use opposite sub-word conventions, which cost a full debugging
round here: `paraformer-zh` marks continuation with a trailing `@@`, so an
unmarked token ends a word; `paraformer-en` marks a word *start* with a leading
`▁` (U+2581), so an unmarked token *continues* a word. Reading the English
vocab with the zh rule split `recirculation` into `rec ir rc ulation` and made
the English checkpoint look identical to the Mandarin one. `pf_detect_word_marker`
now counts both markers over the whole vocab at load - the tables are 0-vs-
thousands, so the winner is unambiguous - and both decode paths share one
`pf_append_token`. Hotword files do not transfer between checkpoints: the
English vocab has `▁set` and no `@@` tokens, so a zh profile silently applies
nothing.

Reproduce with `tools/compare_checkpoints.py`, which reads `voice_languages.tsv`
and scores both checkpoints on the same wav files with the same rule:

```sh
python3 tools/compare_checkpoints.py \
    --zh "$PF_MODELS/paraformer-zh-q8_0.gguf" \
    --en "$PF_MODELS/paraformer-en-q8_0.gguf" \
    --dir /tmp/voice-Eddy --dir /tmp/voice-Rocko --out /tmp/ckpt-ab.json
```

`tools/sensevoice_baseline.py` is a third, non-Paraformer arm (SAN-M + CTC,
25055-token multilingual vocab) used to check that a result is a property of the
audio rather than of one architecture. It needs `onnxruntime` in the venv:

```sh
SV="$PF_CACHE/sensevoice-original"          # or wherever the ONNX export lives
"$PF_CACHE/venv/bin/python" \
    tools/sensevoice_baseline.py --onnx "$SV/model.int8.onnx" \
    --tokens "$SV/tokens.txt" --lang en \
    --refs tools/cabin_commands.tsv /tmp/voice-Eddy/*_en*.wav
```

`PF_CACHE` is the scratch root from [NOTES-REPRODUCE.md](NOTES-REPRODUCE.md); any
directory holding the venv and the SenseVoice export works.

See `docs/backend/snapdragon/paraformer-htp-adaptation-design.md` for the
stage-split, length-bucket, overlap/stitch and bilingual plan.
