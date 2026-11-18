# Paraformer production work log

The product receives complete 16 kHz VAD segments and returns final Mandarin and English transcripts. The target device is SM6650, with up to four concurrent requests and no required partial display. The existing Q8_0 HTP + CPU configuration is the reference. Compare changes against endpoint-to-final P95 as well as failures, transcript and command-field accuracy, and device resource use.

## Paths and reproducibility

Nothing in this file depends on the machine it was written on. Three shell variables stand in for the paths a reader will use; set them once in the shell that runs these commands and every snippet below works unchanged.

| Variable | Meaning | Example |
|---|---|---|
| `PF_MODELS` | Directory holding the `.gguf` weights and the `tingting/`, `voices/` fixture trees | `~/models/paraformer` |
| `PF_CACHE` | Scratch/tooling root: python venv, ONNX reference checkouts | `~/pf-cache` |
| `PF_RUN` | Run directory for one recorded experiment | `~/pf-runs/quality-1` |

```sh
export PF_MODELS="$HOME/models/paraformer"
export PF_CACHE="$HOME/pf-cache"
export PF_RUN="$HOME/pf-runs/quality-1"
```

Get the weights with [`tools/fetch_models.py`](README.md#weights); it verifies sha256 and is the only supported way to obtain a byte-exact copy. Build the binaries as described in [the README](README.md). Fixtures are regenerated, not shipped - see [Fixture provenance](#fixture-provenance).

Some sections cite an archive directory named like `$PF_RUN/<stage>.<nonce>` (`lifecycle.Ojlk2S`, `quality-funasr.7dHKEX`, ...). Those directories exist **only on the authoring machine**. They are cited so a reviewer can see which run a number came from, not because a reader is expected to open them. Every command needed to re-derive the numbers is inline in the section, and `reproduce.sh` runs the accuracy gate end to end. Treat a missing archive as normal.

## Request lifecycle and input boundaries

Baseline source `7d7590cdc` reproduced two WebSocket failures on macOS CPU. An empty commit did not return an error. Two consecutive accepted commits on the same connection returned only the second result.

Each admitted commit owns an item ID and returns its own terminal result or error while the connection stays open. `clear` drops uncommitted audio and invalidates previews. A full queue returns an error before acknowledging a commit and leaves buffered audio available for retry. Queued work checks for disconnect or an obsolete preview before using an inference engine. Previews only append text when the new hypothesis extends the already published prefix.

All server inputs now enforce a 30-second segment limit. TCP validates sample alignment before reading into its float buffer, caps connections at 16, and applies 10 s receive and 5 s send inactivity timeouts. HTTP bounds the complete body at 2 MiB. Raw float PCM rejects non-finite samples on both transports. WAV parsing checks the PCM format tag and complete samples, and WS validates event field types before converting them.

### Replay on the host

```sh
cmake --build build-x-asr --target llama-paraformer-api llama-paraformer-server -j 6
PF_MODELS="${PF_MODELS:-$HOME/models/paraformer}"
PF_WAV="$PF_MODELS/tingting/hvac_zh01.wav"
build-x-asr/bin/llama-paraformer-api "$PF_MODELS/paraformer-zh-q8_0.gguf" \
    --bind 127.0.0.1 --port 18785 --engines 1 --threads 2 --no-partials --warmup "$PF_WAV"
```

In a second terminal, run the smoke test with Python and the `websockets` package installed.

```sh
PF_WAV="$PF_MODELS/tingting/hvac_zh01.wav"
python3 examples/paraformer/tools/smoke_production.py --port 18785 --wav "$PF_WAV"
```

The macOS ARM64 CPU run passed all 18 checks with VAD, speaker scoring, and previews disabled. A separate loopback TCP run rejected unaligned and non-finite audio, then returned a nonempty transcript for the same WAV.

### Android verification

Cross-build both server binaries using [the Android build](README.md#openai-compatible-api) with the existing `build-android-xasr` release configuration and `XASR_SPEAKER=ON`. Run the API binary with the same options as the host replay, using device paths for the model and WAV. Add `--vad-model onnx/model/silero_vad.onnx`, include `onnx/bin` in `LD_LIBRARY_PATH`, and forward the loopback port with ADB. Pass `--expect-vad` to the smoke test.

An isolated API instance on SM6650 passed 20/20 checks with one CPU engine, two threads, VAD enabled, and previews and speaker scoring disabled. This verifies the ARM64 implementation and runs separately from the installed HTP service. The baseline observations, host and Android raw output, build logs, and source, model and binary hashes are archived in `$PF_RUN/lifecycle.Ojlk2S`.

## Backend readiness and inference status

The API and framed TCP service start listening only after their configured engines initialize and warm up. The framed benchmark service also retains `--no-warmup` for measuring cold starts. When CIF emits no tokens during a silence warmup, a zero embedding exercises the decoder graph. Warmup verifies that both stages execute some compute on the selected backend. `/health` keeps the engine count and reports `backends` with the requested and actual device and compute-node counts for both warmup graphs. Views and reshapes are excluded. Fbank, CIF and output conversion also run on the host and are outside these graph totals.

An unavailable requested accelerator or failed warmup exits before listening by default. `--allow-cpu-fallback` opts into a CPU startup retry. The API reports HTTP 200 and `status: degraded` in that case, and `/metrics.json` reports `cpu_fallback` as true (`/metrics` exposes it as `pf_cpu_fallback 1`). A valid no-token decode returns empty text successfully. Failed graph allocation or compute returns an inference error through plain, timestamped, and windowed decoding. The graph cache drops failed stages instead of reusing them. The temporary source GGUF bytes are released after the backend buffers are materialized.

The host run verified strict failure with missing HTP, strict failure with a sub-frame warmup WAV, and successful CPU fallback with `degraded` and `requested=HTP0:0 actual=CPU`. Both ordinary CPU and fallback CPU passed 20 smoke checks. With the same WAV in a separate SM6650 instance, HTP performed 1454/1504 encoder and 559/575 decoder compute nodes. All 66 nodes outside HTP were `FLASH_ATTN_EXT`. Both engines warmed up and the mixed instance passed 20 smoke checks, including empty text from a valid sub-frame PCM request. This is a placement and API check at the warmup shape. Compare attention paths within the same API binary before changing the serving default.

To replay, use the host and Android builds and model above. On macOS run with `--device HTP0:0` and expect startup exit 3, then add `--allow-cpu-fallback` and run the smoke script with `--expect-backend CPU --expect-degraded`. On SM6650 run the HTP launch from the [API section](README.md#openai-compatible-api) with `--engines 2 --threads 2 --no-partials` and a real speech warmup WAV. Pass `--expect-backend HTP0:0` to the smoke script. The exact launchers, source and device hashes, and raw results are in `$PF_RUN/readiness.hzWCkj`.

The Hexagon backend can abort the process on a DSP error during synchronization or tensor readback. Runtime fault isolation and retry still need a recovery path that can survive termination of the accelerator worker. The startup fallback above covers initialization and recoverable warmup failures.

## Hotword rewrite timestamps

Each decoder choice keeps its source CIF position. A length-changing rewrite keeps both endpoints of the original span. Its interior positions interpolate between the original CIF rows. Choices after the rewrite retain their original rows for both scoring and timestamps. Rewritten tokens use the acoustic probability of the replacement at the nearest original row. These timings describe CIF anchors, so precise acoustic word boundaries still require alignment against the reference. CLI token times, the centisecond C API, and REST word spans derive from the same mapped records. Windowed decoding adds the global offset to both ends of each span.

The Q8_0 shipping profile changed `likes` from two to three pieces within 720-1020 ms and `footwall` from five to four pieces within 300-720 ms, on macOS CPU and SM6650 HTP. The unchanged suffixes kept their original timestamps. The final `light` of `light_en03.wav` stays at 2340 ms, where it previously flipped to 0 ms. Plain and timed CLI transcripts agreed on all 32 Tingting fixtures. A 22 s fixture with six commands and silence exercised the model's windowed path. On the SM6650 REST API with server VAD off, 52 ordered word spans ended at 20.060 s.

```sh
PF_MODELS="${PF_MODELS:-$HOME/models/paraformer}"
PF_WAVS="$PF_MODELS/tingting"
PF_MODEL="$PF_MODELS/paraformer-zh-q8_0.gguf"
PF_PROFILE=examples/paraformer/hotwords_vehicle_replace.txt
build-x-asr/bin/llama-paraformer "$PF_MODEL" "$PF_WAVS/light_en03.wav" "$PF_WAVS/light_en04.wav" \
    --threads 2 --timestamps > rewrite-off.tsv
build-x-asr/bin/llama-paraformer "$PF_MODEL" "$PF_WAVS/light_en03.wav" "$PF_WAVS/light_en04.wav" \
    --threads 2 --timestamps --hotwords "$PF_PROFILE" > rewrite-on.tsv
python3 examples/paraformer/tools/check_rewrite_times.py rewrite-off.tsv rewrite-on.tsv
```

The run archive holds the long fixture generator, isolated Android replay and REST assertions, both CLI TSV arms, REST JSON, and artifact hashes under `$PF_RUN/rewrite-times.AkZMgI`.

## Final request VAD and long segments

WAV, raw PCM and WebSocket final requests now share a single admission and decode path. Digital silence returns without model inference even with server VAD off. When enabled, each engine owns one validated VAD detector and resets it before each analysis. Ordinary short speech only pays the RMS check. Long clips group detected spans, retain up to 100 ms of nonoverlapping edge context and send each actual slice to the model, including a single detected slice. WAV verbose responses retain words for every slice and offset them to the original audio. Silence responses preserve the requested format. Each admitted segment occupies one worker for all its slices. The impact of long requests on short-command queueing remains part of the scheduling tests. The framed TCP benchmark server uses the model's native windows.

On SM6650, one HTP engine with Silero VAD passed 22 smoke checks. WAV, raw PCM and WS matched on five fixture replays. The 3 s command stayed one slice. A 22 s clip with a single late command returned one cropped slice and 11 words spanning 15.084-17.484 s in the original input. A 22 s clip with several commands returned two slices and 54 words spanning 0.076-20.052 s. Repeating the late command after the multi-slice clip gave identical word times. The long-clip VAD step took about 140 ms in the first replay. The same replay measured under 0.1 ms for the short command's energy check. macOS CPU with server VAD off passed 20 smoke checks, including digital silence.

Run the Android API with `--vad-model onnx/model/silero_vad.onnx --engines 1 --threads 2 --no-partials`, the Q8_0 model and the vehicle rewrite profile. Forward the loopback port and run the Python verifier with `websockets` installed.

```sh
PF_WAVS="$PF_MODELS/tingting"
python3 examples/paraformer/tools/check_vad_ingress.py --port 18788 \
    --wav "$PF_WAVS/light_en03.wav" "$PF_WAVS/light_en04.wav" --out /tmp/pf-vad-replay
```

The build log, device hashes, launcher, fixture generators and per-transport outputs are archived in `$PF_RUN/vad-ingress.ARo9fu`.

## Bounded speaker scoring

The API runs CAM++ scoring on one CPU thread and admits at most one scorer task, including its active task. It tries to schedule scoring only when a final request is admitted, the ASR queue and workers are idle, and the scorer slot is free. It copies at most the first 1.5 s by default. Final replies inspect the result once at ASR completion. A slower scorer can finish independently as the ASR result is delivered. The requester does not receive a later update for that result.

Both REST endpoints and WS final responses distinguish `disabled`, `assessed`, and `not_assessed`. `speaker_ok` exists only for `assessed`. The not-assessed reasons are `busy`, `in_progress`, `no_audio_in_prefix`, `no_embedding`, and `silence`. `/metrics.json` reports `speaker_started`, `speaker_assessed` and `speaker_deferred`. Supplying a speaker configuration with unavailable model support or enrollment fails before listen. The raw PCM endpoint follows the same policy as WAV and WS.

On SM6650 with the same binary on both arms, six rounds of four aligned WS requests completed with identical transcripts in 24/24 pairs. Enabled scoring started once per round. Six responses carried a score and 18 returned `not_assessed` with `busy`. Endpoint-to-final P50/P95 was 527/724 ms with scoring off and 547/729 ms with scoring on. This was one short off/on comparison. The enabled configuration passed 26 smoke checks. A sub-frame request returned `not_assessed` with `in_progress`, and WS silence returned `not_assessed` with `silence`. Synthetic enrollment and independent synthetic speech yielded example scores of 0.926 and 0.339. These samples establish response semantics and exercise the scorer. Cabin false accept and false reject rates are still open.

Build Android with `XASR_SPEAKER=ON` and launch [the API](README.md#speaker-gate-optional) with two engines (`HTP0:0` + CPU), two inference threads, previews off and server VAD off. Replay with the same 2.32 s mono fixture and no speaker flags for the off arm. Enable `--speaker-model` and `--enroll-dir` for the on arm. Install `websockets` for both commands.

```sh
PF_WAV="$PF_MODELS/tingting/hvac_zh01.wav"
python3 examples/paraformer/tools/smoke_production.py --port 18789 --wav "$PF_WAV" --expect-backend HTP0:0 --expect-gate
python3 examples/paraformer/tools/bench_ws_realtime.py --url ws://127.0.0.1:18789/v1/realtime \
    --metrics http://127.0.0.1:18789/metrics.json --wav "$PF_WAV" --sessions 4 --rounds 6 --out speaker-on.jsonl
```

The Android launcher, synthetic enrollment fixture identifiers and hashes, raw response JSON, per-commit JSONL, and counters before and after each arm are in `$PF_RUN/speaker.5xc2A3`.

## HTP attention placement and latency

`--no-htp-flash-attn` selects the existing QK matmul, softmax and value matmul graph for HTP. The flag keeps CPU workers on their existing attention graph. `/health` reports the attention choice per engine. Both REST endpoints and WS timings expose the worker backend for each request.

The same API binary and Q8_0 model ran on SM6650 with the vehicle rewrite profile, two inference threads per worker, and previews, server VAD, and speaker scoring off. The recognition check ran one HTP engine per arm on the 32 Tingting clips. The scheduling check used two engines (HTP + CPU) and restarted in flash, decomposed, decomposed, flash order. For each start it used one warmup request and two rounds per layout. All workloads used paced 20 ms WebSocket packets.

| HTP attention | Encoder ops on HTP | Decoder ops on HTP | zh CER | en WER | Median inference, 32 HTP requests |
|---|---:|---:|---:|---:|---:|
| Flash | 1454/1504 | 559/575 | 0.60% | 7.63% | 293.5 ms |
| Decomposed | 1704/1704 | 639/639 | 0.60% | 6.87% | 288.5 ms |

Placement counts describe the warmup graphs. The previous CPU placements were all `FLASH_ATTN_EXT`. On the decomposed path, `light_en02` changed from `reading leg` to the reference `reading light`. The other 31 fixture transcripts matched exactly.

| Paced workload | Requests per arm | Flash P50/P95 | Decomposed P50/P95 |
|---|---:|---:|---:|
| Single 2.322 s command | 4 | 348/383 ms | 328/371 ms |
| Four aligned commands | 16 | 508/731 ms | 468/730 ms |
| Four staggered starts | 16 | 368/447 ms | 349/477 ms |
| One 10 s clip and three short commands | 16 | 667/1462 ms | 647/1452 ms |
| Short requests within the mixed workload | 12 | 565/785 ms | 545/825 ms |

Latency is measured from client commit to final, and the maximum lateness of the client commit versus its scheduled endpoint was 2.31 ms. All 52 benchmark transcripts matched per arm. In the aligned workload the median HTP worker inference fell from 314 to 282 ms. Client P95 stayed close there and increased in the staggered and short-with-long workloads. Keep the default graph and expose decomposition as an opt-in HTP candidate. The short samples motivate backend-aware scheduling before a default switch.

The pre/post snapshots showed about 415-425 MiB RSS, zero process swap, and Android thermal status 0 for both arms. These are short-run process snapshots. Sustained thermal and system contention remain open.

Run the two Android API arms with `--device HTP0:0 --engines 2 --threads 2 --no-partials` and the same warmup WAV. Add `--no-htp-flash-attn` for the decomposed arm. Forward the loopback port and run the paced client.

```sh
PF_CACHE="${PF_CACHE:-$HOME/pf-cache}"
PF_WAVS="$PF_MODELS/tingting"
python3 examples/paraformer/tools/bench_ws_realtime.py --url ws://127.0.0.1:18791/v1/realtime \
    --metrics http://127.0.0.1:18791/metrics.json --sessions 4 --rounds 2 \
    --wavs "$PF_CACHE/cabin-asr-probe/long-perf/zh-10.wav" "$PF_WAVS/hvac_zh01.wav" \
    "$PF_WAVS/light_en01.wav" "$PF_WAVS/light_en03.wav" --start-offsets-ms 0 7700 8050 7570 \
    --out mixed.jsonl
```

The exact launcher, quality replay, raw requests, backend placement, hashes, memory and thermal snapshots are archived in `$PF_RUN/attention-ab.rPnRkA`.

## HTP encoder width-two probe

`llama-paraformer --bench-encoder-pair REPS` compares two width-one encoder executions with one width-two execution. Both arms use the same weights, fbank features and sinusoidal positions. Width two flattens the time and batch axes only for the shared-weight projections and FFN. Attention and FSMN retain a separate axis for each clip. The probe warms both graphs twice, alternates execution order and swaps the clips across lanes. Wall time covers input upload, execution and output readback. Features, graph allocation, the CIF predictor and the decoder are outside the timed region.

The SM6650 run used Q8_0, decomposed attention, two threads and two distinct clips per width. It dispatched all 1704 compute nodes to HTP in one scheduler split at both widths and lengths. All encoder output values matched exactly, including the swapped lanes. The macOS CPU control also matched exactly. With the vehicle rewrite profile applied, the ordinary width-one CPU CLI returned the same tokens, text and timestamps as the saved baseline on 32/32 clips.

| Equal audio length per lane | LFR frames | Two width-one executions P50 | One width-two execution P50 | Pair time saved |
|---|---:|---:|---:|---:|
| 2.70 s | 45 | 483.24 ms | 470.27 ms | 12.97 ms (2.7%) |
| 5.40 s | 90 | 944.05 ms | 929.22 ms | 14.83 ms (1.6%) |

Keep this as a CLI benchmark. The pair saves about 13-15 ms when both rows are already available. The first item in a serial pair can finish its encoder stage much earlier than the batched pair, which matters for staggered commits. Service scheduling continues to submit one segment per engine call.

```sh
PF_CACHE="${PF_CACHE:-$HOME/pf-cache}"
PF_OUT="${PF_RUN:-$HOME/pf-runs}/encoder-b2"
python3 "$PF_OUT/make_fixtures.py" "$PF_MODELS/tingting" "$PF_OUT"
build-x-asr/bin/llama-paraformer "$PF_MODELS/paraformer-zh-q8_0.gguf" \
    "$PF_OUT/short-a.wav" "$PF_OUT/short-b.wav" --threads 2 --bench-encoder-pair 6
bash "$PF_OUT/android.sh" "$PF_OUT"
```

The fixture generator zero-pads `hvac_zh01.wav` and `light_en03.wav` to 43,200 samples and doubles each padded sequence for the long arm. The run directory holds input and binary hashes and raw logs from six repetitions per host shape and ten repetitions per device shape.

## HTTP and WebSocket admission and deadlines

The API caps its connection pool at 12 active socket handlers with up to eight sockets queued. Four WebSocket sessions can be active. A fifth upgrade is closed with code 1008. Each session admits up to four pending commits. `--queue` and `--queue-mib` bound ASR requests waiting for an engine by count and allocated PCM capacity, with defaults of 64 and 8 MiB. A rejected WS commit retains the audio for retry. Explicit `clear` releases the buffer allocation. HTTP returns 429 when a request exceeds the admission budget and 400 for an invalid segment.

`--deadline-ms` defaults to 30,000 ms and starts at final request submission, after HTTP body intake or WS COMMIT. REST returns 504 with `deadline_exceeded`. A WS commit keeps its item ID and receives one terminal `error` with the same code. The WS poll loop checks for expired items every 40 ms when idle. The worker skips expired queued work and drops the result of a completed decode if it missed the deadline. A model call already running continues to hold its engine until it returns. `--timeout-ms` continues to cap the wait for an engine and returns 503 when that limit is reached first.

```sh
PF_MODELS="${PF_MODELS:-$HOME/models/paraformer}"
PF_WAV="$PF_MODELS/tingting/hvac_zh01.wav"
build-x-asr/bin/llama-paraformer-api "$PF_MODELS/paraformer-zh-q8_0.gguf" \
    --bind 127.0.0.1 --port 18797 --engines 1 --threads 2 --no-partials --warmup "$PF_WAV" \
    --queue-mib 1 --deadline-ms 20
# From a second terminal with websockets installed, or against the forwarded device port
python3 examples/paraformer/tools/check_admission.py --port 18797 --wav "$PF_WAV"
```

The host CPU and SM6650 HTP both rejected an over-budget PCM request, rejected a fifth session, let a WS client retry a rejected commit, returned three deadline terminal responses across WAV, raw PCM and WS, suppressed a late duplicate result, and accepted a reconnect. Each run ended with zero queued bytes and zero in-flight requests. The 20 ms profile is a test configuration. Launchers, hashes and raw responses are in `$PF_RUN/admission.iALkbe`.

Disconnect releases a session's uncommitted PCM and removes its waiting jobs from the queue. Run the same checker with `--disconnect-only` against a single engine using the default queue and deadline. On the host and SM6650, filling the per-session pending limit led to one overload reply and four cancellations. Closing the queued socket returned `queue_bytes` to zero before the active request finished. In a separate Android run, four of 24 simultaneous incomplete HTTP requests were dropped and `/health` returned 200 after closing the sockets. The final device checks and socket replay are in the routing run directory.

## SM6650 replay baseline

The post-admission API ran for 307 s on the attached SM6650. It used Q8_0, one HTP engine using the default attention path, one CPU engine, two threads per engine and the vehicle rewrite profile. Previews, server VAD and speaker scoring were disabled. Clients sent 20 ms PCM packets with real-time pacing, opened new sessions per round and measured from COMMIT to terminal completion. The staggered arm used four starts. The mixed arm brought a 10 s clip and three short clips to roughly the same endpoint.

| Workload | Requests | P50 | P95 | P99 | Maximum |
|---|---:|---:|---:|---:|---:|
| Four aligned 2.32 s clips | 180 | 516 ms | 709 ms | 739 ms | 771 ms |
| Four staggered clips | 40 | 366 ms | 445 ms | 447 ms | 448 ms |
| Four mixed lengths | 40 | 669 ms | 1453 ms | 1478 ms | 1494 ms |
| Short clips within mixed lengths | 30 | 565 ms | 770 ms | 796 ms | 807 ms |

All 260 final requests succeeded, with zero rejects, deadlines or inference failures. Between warmup and the final snapshot, RSS ranged from 419 to 426 MiB. `VmHWM` records a 642 MiB peak including startup. Process swap stayed at zero and Android reported thermal status 0 in all 22 snapshots. This soak establishes a baseline for the tested device and load. Test duration was five minutes and there was no separately injected CPU contention.

The isolated build and launch commands, metrics, per-request JSONL, parts of `/proc`, thermal outputs and artifact hashes are in `$PF_RUN/admission.iALkbe`. Re-run with `bash "$PF_OUT/device.sh" "$PF_OUT"` from the repo, with `PF_OUT` set to that run directory. Device scripts use temporary paths and an ADB loopback forward.

## Backend scheduling on SM6650

`--prefer-idle-htp` lets an idle HTP worker take the next request from the shared queue. A CPU worker can take the next request as soon as HTP has a request of its own. The policy waits for no batch and preempts no running inference. It uses worker occupancy. Operating-system CPU availability and inference duration prediction are separate inputs for a future routing comparison.

The same Android binary alternated two-worker runs with and without the flag. The same replay also ran one arm per mode with a second CPU worker. All arms used the same Q8_0 weights, default attention graph, two threads per worker, vehicle rewrite profile and 20 ms packets. The table gives COMMIT-to-final P95 in milliseconds.

| Workload | HTP + one CPU, shared | HTP + one CPU, prefer idle HTP | HTP + two CPUs, shared | HTP + two CPUs, prefer idle HTP |
|---|---:|---:|---:|---:|
| Single command | 368 | 329 | 367 | 363 |
| Four aligned | 727 | 726 | 753 | 773 |
| Four staggered | 440 | 447 | 462 | 447 |
| Mixed long and short | 1454 | 1212 | 1529 | 1252 |
| Short requests in mixed arm | 766 | 768 | 786 | 528 |

With two workers, the flag routed all six 10 s clips to HTP. The shared queue sent three to CPU. Favoring idle HTP reduced the mixed P95 by 242 ms. Short-request P95 in that mix stayed within 2 ms. All 228 requests completed, with identical text per workload across arms. For the extra CPU worker, mixed short tail is promising at nine requests per arm. The aligned tail rose and RSS after replay was about 721 MiB with three workers versus 422 MiB with two. Keep `--prefer-idle-htp` available for the two-engine mixed workload. Keep the third worker out of the baseline until repeated under a representative concurrent CPU load and memory budget.

The second ABBA run pinned one busy thread each to CPUs 6 and 7. `/proc/stat` measured 100% busy time on both cores over the test and 10-23% on the remaining cores. Both routing arms completed all 144 requests with matching transcripts. Default versus idle-HTP P95 was 884 versus 833 ms aligned, 520 versus 527 ms staggered, and 1558 versus 1451 ms mixed. P95 for short requests within mixed fell from 1059 to 847 ms. The candidate remains one HTP and one CPU worker with `--prefer-idle-htp`. The test used synthetic CPU contention and thermal status remained 0.

The ABBA launchers, per-request results with `backend`, health, artifact hashes, and resource snapshots are in `$PF_RUN/routing.vjddv2`. Use `bash "$PF_OUT/route.sh" "$PF_OUT"` and `bash "$PF_OUT/pressure.sh" "$PF_OUT"` from the repo with that run directory as `PF_OUT`. The recorded pressure hash is from the phone. The initial routing executable was reconstructed from the archived source version after the manifest file was reused for the pressure run. It matches the logged transfer size. The model and backend libraries were identical in both runs.

### Reordering the queue does not help

Shortest-first dequeue was tried to protect short commands from a long upload and then reverted. The idea was that with two workers, arrival order puts a short command behind a long one and pays a full long decode, while shortest-first finishes the shorts inside one long decode.

On SM6650 with two workers, `--prefer-idle-htp` and a 13.9 s clip against three 2.3-2.7 s clips released together, the median fell from 869 to 690 ms but the tail rose from 1536 to 1825 ms. Per-request rows show why. Short requests improved from 1049 to 364 ms, and the long request paid for it: its queue wait went from 0 to 350-400 ms because three shorts were taken ahead of it, so it finished later than in the FIFO arm. The total work is unchanged, so the median win is a transfer, not a saving, and the tail absorbs the loss.

Ageing was what produced the tail. A long request deferred behind shorts was only taken once it passed the 2000 ms age bound, which is longer than the 1.4 s the decode itself needs.

The conclusion is structural: with two workers and four requests, FIFO completion order matches the work already in flight, and any reorder moves latency between requests rather than removing it. Short commands are not slow in this workload - they take 205-366 ms end to end and the mixed P95 of 1452 ms is the long decode itself. Improving them needs a long request that yields its worker mid-decode, which the engine cannot do today because one context owns the whole window loop. Treat `--queue-order` as unavailable until that changes.

## Process recovery on CPU

[`tools/supervise_api.sh`](tools/supervise_api.sh) runs the API as a child. An abnormal HTP exit triggers a new CPU-only API child on the same port. `--recover-on-cpu` preserves the configured HTP backend in the first engine's `requested` health field, reports `actual: CPU` and marks the process `degraded`. The CPU child loads and warms before listening. It stays on CPU until the service is restarted. Start the supervisor under the device's service manager so the CPU child's own exit is visible to that manager.

Run on the device from a prepared package with the Q8_0 model, the Android binary and its matching ARM64 and Hexagon libraries. Check SHA-256 digests of the copied package before starting. Supply your transport address and warmup file.

```sh
export LD_LIBRARY_PATH="$PWD/lib:$PWD/onnx/bin"
export ADSP_LIBRARY_PATH="$PWD/lib"
export GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPPOLL=1
sh tools/supervise_api.sh ./bin/llama-paraformer-api paraformer-zh-q8_0.gguf HTP0:0 \
    --bind 127.0.0.1 --port 18798 --engines 2 --threads 2 --no-partials --warmup pf-api-warm.wav
```

On SM6650, sending SIGKILL to the isolated HTP child after an acknowledged WS COMMIT closed that connection. Health returned `degraded` with two CPU engines on the same port after 1.57 s. The client opened a new socket, resent audio and received a correctly timed CPU result. The recovered child passed 21/21 smoke checks. The injection tests process death, the supervisor transition and client recovery. DSP faults and hangs require separate injection. A process that hangs needs the service manager's watchdog. Recovering an interrupted WS commit requires the client to retain and resubmit its audio after reconnect. The hashes, injection script, event output and supervisor log are in `$PF_RUN/admission.iALkbe`.

Repeated on SM6650 with `SIGKILL` on the api child: the supervisor logged `exited mode=htp status=137`, started `mode=cpu`, and the new child reported `requested=HTP0:0 actual=CPU` with a full 1504/1504 encoder and 575/575 decoder placement in 329 ms. `/health` returned `degraded` and a plain `POST /v1/audio/transcriptions` still returned the correct text.

### Watchdog scope

`--htp-watchdog-ms` exits the process when a single backend call runs past the limit. It covers backend discovery, engine warmup and one inference, and it exits with status 75 rather than unwinding, because the backend may be parked in FastRPC.

Warmup and discovery take longer than one inference: 1.3 s against 0.52-0.56 s on SM6650 with Q8_0. Using the inference limit there killed the process during start-up, so a limit below about 1.4 s could never start the service at all. Warmup and discovery now use `--htp-warmup-ms`, defaulting to twenty times the inference limit, so the same 400 ms inference limit now starts in 1332 ms and still exits during inference when a single call overruns.

The limit must stay above the slowest single inference at the deployed audio length. A 400 ms limit on a 13.9 s clip exits as designed, because that clip needs about 1.4 s. Warmup does not close the gap: the first inference after warmup took 560 ms against 522 ms steady, so size the limit from steady-state inference plus margin, not from the warmup figure.

## Cabin command accuracy and FunASR

The 96-clip regression uses 32 command texts across window, climate, seat and cabin light functions, synthesized in two voice sets. A further 16 confusable clips include four commands and 12 controls. Another 16 clips mention cabin devices without issuing a command. The shared offline evaluator re-parsed all 96 annotations before scoring. `device` includes the required position and `value` includes the annotated level, number, percentage or color. The command metric requires every annotated clause to match. Potential false activation is output that the offline evaluator maps to an actionable clause. Ship decisions require an evaluation of the actual downstream command parser and extra actions as well.

The upstream CPU reference used the plain FunASR Paraformer checkpoint with VAD and punctuation off. The phone run used the Q8_0 API with vehicle rewrites and 600 ms acoustic punctuation. Keep this postprocessing difference in view when reading command counts. The host comparison includes raw and postprocessed runs of the local engine. The source checkpoint is [FunASR Paraformer on ModelScope](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch).

| Arm | Tingting command | Second voice command | Tingting zh CER / en WER | Second voice zh CER / en WER | False triggers in 28 controls |
|---|---:|---:|---:|---:|---:|
| FunASR CPU, raw | 17/32 | 6/32 | 0.60% / 12.98% | 30.12% / 21.37% | 0/28 |
| Q8_0 CPU, with punctuation | 21/32 | 8/32 | 0.60% / 12.21% | 34.34% / 22.90% | 0/28 |
| Q8_0 CPU, rewrites and punctuation | 26/32 | 9/32 | 0.60% / 6.87% | 34.34% / 18.32% | 0/28 |
| Q8_0 SM6650 HTP, rewrites and punctuation | 26/32 | 9/32 | 0.60% / 7.63% | 34.94% / 19.08% | 0/28 |

That table predates both the mixed-language fixture and the hotword retune, and its `26/32` is row-level rather than the clause-weighted `command` field the current evaluator reports; the two are not comparable, so the numbers below are given separately rather than appended to it.

| Arm (clause-weighted `command`, no punctuation) | 32-clip cabin | 15-clip mixed | Tingting zh CER / en WER |
|---|---:|---:|---:|
| Q8_0, no hotwords | 56.2% | 80.0% | 0.60% / 12.21% |
| Q8_0, §13-14 profile | — | **70.0%** | 0.60% / 8.40% |
| Q8_0, §15 profile as first committed | 65.6% | **100.0%** | 0.60% / 8.40% |
| Q8_0, §15 profile with the two noun rewrites (§32) | **71.9%** | **100.0%** | 0.60% / **6.87%** |
| Q8_0 SM6650 HTP, same profile | **71.9%** | **100.0%** | 0.60% / **6.87%** |

The device row is not borrowed from the host: it was re-measured on SM6650 with `--device HTP0`, and the 47 transcripts behind it are byte-identical to the host ones. `R` fires on the same spans because the logits come from the same quantized weights, so the 2.39 deficit that calibrates the `rear right` margin is unchanged on device.

The §13-14 profile scores *below* no hotwords on mixed speech: its global `1.5 two` gain fires after `to` in `temperature to 二十一度` and inserts a token that the parser reads as 11 degrees. The current profile replaces it with `1.5 l>two`, which is narrow enough to fix `level two` and not fire at the boundary.

The §15 retune initially dropped the two noun rewrites (`footwall`, `likes`) that `hotwords_vehicle_replace.txt` had carried, on the reasoning that they belonged to the deprecated file. They do not carry the poisoned `two` rule, and restoring them is worth 6.3 points of cabin command and 1.5 points of en WER with mixed command and Mandarin unchanged. The lesson is about the file, not the rules: deprecating a profile should move its good rules out, not discard them with the bad one. See design doc §32.

Accuracy is settled at mixed command 100%, Mandarin CER 0.60% and dictation English 6.87%. The remaining six English errors are missing word forms (`heater`, `massage`, `vents`) and multi-token clumps (`driversee peater`) that no logit bias can reach: `R` can rewrite an observed span but not one the model never emitted.

On HTP the first and second voice sets got 37/42 and 20/42 device-position clauses, and 18/18 and 5/18 value clauses. Negation and exclusion cases scored 3/4, 1/4 and 2/3 across the two voice sets and the confusable pack. The second voice exposes a release blocker even for the upstream model. Keep the existing profile as a regression arm and avoid deriving new substitutions from the holdout run.

Real Mandarin replay used 36 non-overlapping conversational clips from six speakers across three AISHELL-5 Dev sessions (DX01C01, 100 ms padding). The transcript has 449 reference characters. On SM6650 the Q8_0 API made 106 errors (23.61% CER) versus 104 (23.16%) for FunASR CPU. The offline grader saw zero modeled command triggers in those 36 clips for either arm. This small sample measures far-field conversational speech and provides negative controls. It contains no annotated bilingual positive vehicle command set. [AISHELL-5 dataset and license](https://www.openslr.org/159/).

The same six-speaker sample exercised the actual Android CAM++ path, with two disjoint enrollment clips per profile and four test clips per speaker. All 144 comparisons returned scores. At the shipping example threshold of 0.55, 5/24 genuine trials rejected (20.8%) and 30/120 impostor trials accepted (25%). At 0.65 those counts were 8/24 and 12/120. Results from shared sessions and clips are correlated and the cohort is small. The scores support retaining speaker assessment as advisory data while the control path is unqualified.

### Voice sweep with synthesized speakers

The second voice set was one speaker. A wider sweep separates "one unlucky voice" from "the model only works for the voice it was tuned on", and needs no recordings.

**Which voices may be scored at all** is decided by [`tools/voice_languages.tsv`](tools/voice_languages.tsv), not by convenience. A voice is only eligible for a language if it is native for that language. The multilingual novelty voices render Mandarin as damaged audio that every recogniser fails on, so a Mandarin clip from Sandy, Eddy, Rocko, Shelley, Reed, Grandma, Grandpa or Flo measures the synthesizer and not the engine. Sinji is a Cantonese voice reading Mandarin and is likewise ineligible. That leaves three Mandarin voices (Tingting, Meijia, Sinji) of which two are usable, and eight native `en_US` voices for English. Every rate in this document is computed on that split.

[`tools/gen_voice_fixtures.py`](tools/gen_voice_fixtures.py) renders the same 32 cabin texts through the macOS voice engine, one output per voice, always 16 kHz mono PCM16. It picks the language variant per row: `say -v Eddy` alone resolves to the English variant, which renders Mandarin as 5 KB of digital silence and still exits 0. [`tools/score_voice.py`](tools/score_voice.py) posts a directory to a running server and reports the same CER and WER the table above uses.

Tingting was re-rendered through the same script as a control. It reproduced the recorded set exactly - 0.60% zh, bit-identical transcripts - so the pipeline is equivalent to the original fixtures and differences between rows are the voice, not the render. English rows are excluded for Tingting because the voice has no `en_US` variant and reads English with a heavy accent; that number measured the voice, not the engine.

| Voice | Native language | Median F0 | zh CER | en WER (zh checkpoint) | en WER (en checkpoint) |
|---|---|---:|---:|---:|---:|
| Tingting | zh_CN | 234 Hz | 0.60% | n/a | n/a |
| Meijia | zh_TW | 220 Hz | 0.00% | n/a | n/a |
| Sinji | zh_HK | 204 Hz | excluded | n/a | n/a |
| Grandpa | en_US | 91 Hz | excluded | 38.93% | 21.37% |
| Sandy | en_US | 237 Hz | excluded | 49.62% | 32.06% |
| Rocko | en_US | 94 Hz | excluded | 33.59% | 22.14% |
| Flo | en_US | 207 Hz | excluded | 40.46% | 25.19% |
| Eddy | en_US | 112 Hz | excluded | 35.88% | 19.08% |
| Shelley | en_US | 156 Hz | excluded | 41.22% | 25.19% |
| Reed | en_US | 100 Hz | excluded | 32.82% | 16.79% |
| Grandma | en_US | 113 Hz | excluded | 43.51% | 31.30% |
| **English total, 8 voices / 1048 words** | | | | **39.50%** | **24.14%** |

The `en WER (en checkpoint)` column is the fixed result of this round; see [English needs the English checkpoint](#english-needs-the-english-checkpoint). Both English columns are the raw CLI arm with no rewrite profile, so they are comparable to each other and to the Whisper row below, and are not comparable to the rewrite-enabled numbers quoted further down.

Mandarin is now measured on two voices instead of eleven, and the two that remain agree: 0.60% and 0.00%. That is the point. The earlier eleven-voice table averaged engine quality with synthesizer damage, so its 19-42% Mandarin rows said nothing about the engine. On eligible audio the engine reads a native Mandarin synthetic voice perfectly.

Pitch and speed do not explain the spread. Tingting and Sandy differ by 3 Hz and 1 point of CER; Meijia and Sinji are both Mandarin voices and differ by 66 points. Slowing Sandy by 15% to the Mandarin voice rate moved 25.90% to 22.89% and speeding Meijia by 15% moved 0.00% to 0.60%, so time scale is worth about 3 points of 26.

### The audio, not the model, is what breaks

The first version of this section concluded the model overfits the tuning voice. That was wrong, and the error is worth recording because the reasoning that produced it looked sound.

FunASR CPU and the Q8_0 engine score these clips almost identically per voice - within 3 points zh on every one. That is not independent evidence: the Q8_0 checkpoint is a conversion of the same FunASR model, so agreement only shows quantisation cost nothing. The two arms fail the same clips because they are one model.

Whisper is a different acoustic model. Its zh CER on the same voices, which is what separates "damaged audio" from "engine cannot read this speaker":

| Voice | Paraformer Q8_0 | Whisper small | Reading |
|---|---:|---:|---|
| Tingting | 0.60% | 13.86% | both read it; engine better |
| Meijia | 0.00% | 9.04% | both read it; engine better |
| Sinji | 65.66% | 21.08% | both fail; accented audio |
| Sandy | 25.90% | 58.43% | audio damaged |
| Eddy | 33.13% | 51.81% | audio damaged |
| Rocko | 34.34% | 48.19% | audio damaged |
| Shelley | 33.73% | 62.65% | audio damaged |
| Reed | 38.55% | 46.39% | audio damaged |
| Grandma | 41.57% | 56.02% | audio damaged |
| Grandpa | 19.28% | 69.28% | audio damaged |
| Flo | 34.34% | 53.01% | audio damaged |

Whisper is worse than this engine on every row, and roughly twice as bad on the multilingual voices. A model that cannot read the clips either is not being out-generalised by one that can. The Mandarin clips from the multilingual voices are damaged at synthesis, and every recogniser inherits that. Whisper is also the weaker model in absolute terms - 13.86% against 0.60% on the clip set it does read - so it is a detector for damaged audio, not a better baseline. Sinji is the one row where both engines fail for a different reason than the others: a Cantonese voice reading Mandarin produces accented audio, so it is excluded from the Mandarin baseline rather than counted as an engine result.

The English side is the opposite case, and this is what the eighth round settled. These eight voices all have `en_US` variants, so the English clips are native speech. Whisper small scores **3.39% WER** on them once numbers are canonicalised (`twenty one` vs `21`, `forty percent` vs `40%`), against **24.14%** for this engine with the correct checkpoint. The audio is clean and a quarter of the words are still wrong. That is a real engine gap, not a fixture artefact.

The number-canonicalisation caveat is not cosmetic. Without it Whisper scores 9.35% instead of 3.39%, because it applies inverse text normalisation that this engine does not: `Set the driver temperature to twenty one degrees` comes back as `... to 21 degrees.`, and every digit costs a word. Scoring the raw strings compares two different output conventions and inflates the ceiling by 6 points. `tools/check_audio_quality.py` canonicalises both arms for the same reason. The direction of the conclusion does not change - the ceiling is low either way - but quoting 9.35% as the audio limit would have made the engine gap look half its real size.

The comparison is reproducible without a second engine build. `tools/check_audio_quality.py` reads whisper's `--output_format tsv` tree alongside any transcript TSV and prints both rates per voice, converting Traditional to Simplified first - without that conversion every Traditional character counts as an error and the whole comparison silently inverts:

```sh
whisper --model small --language zh --task transcribe --output_format tsv \
    --output_dir /tmp/wsp-zh /tmp/voice-sandy/*_zh*.wav
python3 examples/paraformer/tools/check_audio_quality.py \
    --refs examples/paraformer/tools/cabin_commands.tsv \
    --arm whisper=/tmp/wsp-zh --arm paraformer=ours.tsv --lang zh
```

The English side splits differently, and this is the part that matters for release. These voices all have `en_US` variants, so the English clips are native speech. The first version of this section concluded that English command recognition was simply weak, at 32-50% WER. That was also wrong, and wrong in a more interesting way: the engine was running the Mandarin checkpoint (`paraformer-zh`, vocab 8404) on English audio. FunASR publishes a separate English checkpoint. See the next section.

### The English spread is a speaker effect, not a language effect

Before choosing a checkpoint it is worth knowing which variable the 39.50% belongs to. The same 16 English sentences were rendered through Tingting, a Mandarin voice, and scored against the same references with the same function. Tingting is not in the English sweep because it has no `en_US` variant - and that exclusion hid the most informative row in the whole table:

| Arm (identical wav references, `norm_en`, no number canonicalisation) | Tingting (zh voice) | 8 native `en_US` voices |
|---|---:|---:|
| `paraformer-zh` Q8_0 | **12.21%** | 39.50% |
| `paraformer-en` Q8_0 | **4.58%** | 24.14% |

Both checkpoints do dramatically better on English spoken by a Mandarin voice. `paraformer-en` reads the Tingting English clips at 4.58%, which is near-perfect, and the Mandarin checkpoint reads them at 12.21%, which is the number the README used to quote. So:

- English is not hard for these models. English from a **native American voice** is hard for them.
- The defect scales with speaker, not with language: 4.58% -> 24.14% for the English checkpoint is a 5x degradation on the same sentences.
- `paraformer-zh` was trained on 60000 h that is mostly Mandarin, with English in it as *foreign words read by Mandarin speakers*. Accented English is inside its training distribution; General American is closer to out of distribution. That is the mechanism, and it explains why the checkpoint with less English training (8404) beats the one with 50000 h of English (10020) on the Mandarin-accented clips.

This also means the 24.14% baseline is not a floor to optimise against for this product. A Chinese user issuing English cabin commands speaks the same accented English Tingting renders, and `paraformer-en` covers that case at 4.58%.

Three independent arms agree, which is what makes this a measurement rather than a theory: Whisper small (9.35% raw / 3.39% canon on the 8 voices, 6.87% on Tingting), SenseVoiceSmall (35.66% Eddy, 20.16% Tingting, 0% zh after ITN canonicalisation), and both Paraformer checkpoints all move in the same direction with the same magnitude. A defect that only one engine showed would be an engine bug; one that moves all three is a property of the audio.

`tools/sensevoice_baseline.py` is the SenseVoice arm. It is a second architecture (SAN-M encoder + CTC, 25055-token multilingual vocab) rather than another Paraformer, so its agreement is not a shared-checkpoint artefact:

```sh
VENV="$PF_CACHE/venv/bin/python"
SV="$PF_CACHE/sensevoice-original"
$VENV examples/paraformer/tools/sensevoice_baseline.py \
    --onnx "$SV/model.int8.onnx" --tokens "$SV/tokens.txt" --lang en \
    --refs examples/paraformer/tools/cabin_commands.tsv \
    $PF_MODELS/voices/eddy/*_en*.wav
```

### Bilingual in one checkpoint: what the data says

The requirement is a single model, Mandarin first, that survives a sentence mixing both languages. Two candidates exist and they fail differently:

| | `paraformer-zh` | `paraformer-en` | SenseVoiceSmall |
|---|---|---|---|
| Vocab / markers | 8404, `@@` | 10020, `▁` | 25055, `▁` |
| Mandarin CER | **0.60%** clean | unusable (pinyin) | **0%** after ITN canon |
| English WER, zh-voice speaker | 12.21% | 4.58% | 20.16% |
| English WER, native US speaker | 39.50% (**6.87%** with shipped profile) | 24.14% | 35.66% |
| Mixed sentence | **works** | collapses to pinyin | works |
| Size int8/Q8_0 | 226 MiB | 228 MiB | 228 MiB |

Measured on 15 mixed sentences rendered through a Mandarin voice (`tools/cabin_mixed.tsv`), the choice is not close. `paraformer-en` fed `打开 front left window` returns `dark high front left window`, and `把氛围灯调成 warm white` returns `bafeng wei de diaocheng born white` - every Mandarin syllable becomes Latin. It cannot be the product model no matter how good its English is, because cabin commands in Chinese are the primary case.

`paraformer-zh` handles the mixed case well: `打开 front left window`, `关闭 seat heater`, `把氛围灯调成 warm white`, `空调调到 twenty two degrees` all come back correct, with the Mandarin part exactly right. Its mixed failures were `rear right` -> `we are right` and `footwell` -> `footal`, both of which the hotword profile now fixes; measured mixed command field is 100% on the 15-clip fixture. `rear right` is worth a note because it was wrongly written off as a vocabulary gap: the vocabulary has no `rear` token, but a pure-English `turn on the rear right reading light` decodes perfectly as `re@@ ar`, so the defect is a zh-to-Latin boundary effect that weakens the first English word, not a missing word.

**So the answer to "is there one model for both languages" is: `paraformer-zh` is the one, and it already is what ships.** `paraformer-en` is a specialist for English-only sessions with native-accented speech; `paraformer-zh` plus a cabin hotword profile covers Mandarin, mixed speech, and Mandarin-accented English. Switching to `paraformer-en` by language would break exactly the mixed case the product needs.

Scoring caveat for mixed speech, which the table above exposes: character CER cannot fairly score `导航到 Beijing 机场` decoded as `导航到北京机场`. The transcription is semantically right and the reference is written in Latin script, so `norm_zh` charges 7 characters for a correct answer. Any mixed-language regression gate needs to score the command fields (window/seat/HVAC target and value) rather than characters, or it will reject correct output. The mixed fixture records its text with the script the speaker actually used, so `导航到 Beijing 机场` stays Latin in the reference and the number is honest about being a literal comparison.

### English needs the English checkpoint

FunASR ships two large Paraformer checkpoints that differ only in training data and vocabulary:

| | `paraformer-zh` | `paraformer-en` |
|---|---|---|
| ModelScope id | `speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | `speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020` |
| Vocabulary | 8404 | 10020 |
| Training data | 60000 h, Mandarin + English | 50000 h, English only |
| Sub-word marker | trailing `@@` | leading `▁` (U+2581) |
| Q8_0 size | 226 MiB | 228 MiB |

Both are the same architecture (220M, 49 encoder blocks, 16 NAR decoder blocks, d_model 512), so the engine loads either one with no code change: the vocabulary size comes from GGUF metadata. `tools/compare_checkpoints.py` scores both on the same wav files with the same rule, and the result on 8 voices / 1048 words is:

| Checkpoint | zh CER (2 voices) | en WER (8 voices) |
|---|---:|---:|
| `paraformer-zh` Q8_0 | 0.60% / 0.00% | 39.50% |
| `paraformer-en` Q8_0 | n/a | **24.14%** |

That is a 15.4 point absolute and 39% relative reduction, measured at matched quantisation (both Q8_0, both 226-228 MiB) so the comparison does not mix in precision. The two checkpoints are complementary, not interchangeable: the English checkpoint fed Mandarin audio emits pinyin-like Latin text (`把左前车窗打开一半` becomes `buto qian chehechuang da kai yiban`), so a bilingual product needs both and a way to choose between them.

**The first attempt at this comparison produced no improvement at all, and the reason was a bug in this engine, not a property of the checkpoints.** The token assembler only understood the zh convention, so it stripped `@@` and inserted a space between Latin runs. Fed the English vocabulary - where a leading `▁` marks a word start and its absence marks continuation - it scored `▁rec` `ir` `rc` `ulation` as four separate words instead of `recirculation`, and did the same to `ventilation`, `conditioning`, `ambient` and every other multi-piece word. The English checkpoint therefore looked identical to the Mandarin one (40.17% vs 40.08%) while actually being much better. Two conventions, opposite meanings for an unmarked token:

- zh vocab: no trailing `@@` means the token *ends* a word.
- en vocab: no leading `▁` means the token *continues* a word.

The checkpoints are unambiguous about which one applies. In the zh vocab `▁` appears 0 times; in the en vocab `@@` appears 0 times. `pf_detect_word_marker` counts both over the whole table at load and records the winner, so a new checkpoint needs no flag and cannot be silently scored with the wrong rule. Both the single-utterance and long-audio paths call the same `pf_append_token`, which is what makes the bug impossible to reintroduce in one path only - the previous version had the rule copy-pasted in two places and the long-audio copy would have drifted on the next change.

### What is left of the English error

The residual errors are systematic, not random, and they cluster on out-of-vocabulary nouns. The table lists both checkpoints' behaviour, because the same word fails for the same reason on either one - what differs is whether the shipped profile can reach it:

| Reference | Typical hypothesis | Status with the shipped profile |
|---|---|---|
| `ambient` | `emmy and`, `amity and`, `empty and` | still failing; not a token in either vocab |
| `seat heater` | `seat either`, `seat theater`, `seat here` | still failing; `heater` absent, `theater` is the nearest token |
| `footwell` | `foot well`, `foot will`, `football` | **fixed** by `R 5.0 f@@+oo@@+t@@+w@@+all => f@@+oo@@+t@@+well` |
| `recirculation` | `recirrculation`, `recy`, `circulation` | present but rarely chosen; no rule |
| `rear right` | `real right`, `we are right` | **fixed** on mixed input by the §15 span rewrite |
| `lights` | `likes` | **fixed** by `R 5.0 li@@+kes => li@@+gh@@+ts` |
| `Set` (sentence-initial) | `said`, `send`, `sets` | fixed on the zh checkpoint by `3.2 set` |

The split is now sharp: everything that could be expressed as a **span rewrite over tokens the model actually emits** has been fixed, and what remains is vocabulary the model does not have. `ambient`, `heater` and `massage` are absent from the 8404-entry vocabulary, and `driversee peater` is a clump that never appears as the right sequence, so `R` has nothing to match. Decode-time bias cannot create a token that does not exist; this needs either an in-domain fine-tune or a proper-noun post-processor. Both checkpoints have the same gap for the same words, so this is not a checkpoint-selection problem and no amount of profile tuning will clear it.

- **`Set`/`said` on the English checkpoint** is a whole-word confusion with a measured gap of 0.62-1.12 logits over the 48 clips that start with "Set" (the model prefers `said` on exactly half of them). A `PF_HOTWORDS` entry of `3.0 ▁set` flips it and moves that corpus from 24.14% to 23.57%, but it also over-fires once: `turn on recirculation and set the fan` becomes `... set set fan`, because the gain applies at every CIF position rather than only word-initially. This measurement belongs to the *en* checkpoint only - `▁set` is a 10020-vocab token and does not exist in the shipped 8404 table - so it is **not** the rule we ship. On the zh checkpoint the same confusion is fixed by `3.2 set` (see the table above), which does apply, because `set` is a single token there. Two checkpoints, two different tokens for the same word, two different rules.
- Note the contrast with `footwell`, which *looked* like the same class of problem and was not. The model had already emitted the right prefix `f@@|oo@@|t@@`; only the final piece was wrong (`w@@|all` for `well`, a 2.89 logit deficit). That is a span rewrite, not a missing word - the fix was `R 5.0 f@@+oo@@+t@@+w@@+all => f@@+oo@@+t@@+well`. §14 concluded "token count differs, bias cannot fix it" from the whole-word form and applied it to the tail, which was the wrong object. The test that separates the two cases: dump the tokens (`PF_DUMP_TOKENS`) and ask whether the *prefix* is already correct.

Hotword files are **not portable across checkpoints**. `hotwords_vehicle.txt` uses `set` (single token in 8404) and `se@@>at` (an `@@`-convention prefix); neither exists in the 10020 vocab, which has `▁set` and no `@@` tokens at all. Loading the zh file against the en checkpoint logs `hotword token not in vocab: 'two'` and silently applies nothing. The bias profile has to be regenerated per checkpoint from that checkpoint's own logit gaps.

TTS still bounds what any of this measures. There is no far field, no road noise, no microphone colouration and no accent, so the English numbers are a floor on speaker sensitivity, not a ceiling on deployment error. Real recorded speech is still required for a release threshold.

For Mandarin, the fallback is unchanged and short: use a Mandarin voice for fixtures. Meijia at 0.00% and Tingting at 0.60% show the engine reads a native Mandarin synthetic voice perfectly, so a broken clip set measures the TTS, not the engine. There is nothing to fix on the Mandarin side at 0.60%.

For English, the fallback is now two things and neither is a decode-time trick. The checkpoint swap is the fix that works today; it costs a second 228 MiB weight file and a way to route each utterance, and it removes 15.4 points. Everything above the remaining 24% sits in nouns that are absent from the 10020 vocabulary, which no runtime knob can repair. The options in cost order are a cabin-domain proper-noun post-processor, in-domain fine-tuning, or accepting 24% and adding a confirmation prompt for commands the confidence cannot separate.

The shipping path is not the English checkpoint, though, and its English number is better than the 24% above. On the shipped `paraformer-zh` plus `hotwords_vehicle.txt`, the `tingting` dictation fixture reads **6.87%** en WER against 12.21% without the profile, with zh CER bit-identical at 0.60% and mixed command at 100%. That is the number that matters for this product, because Mandarin is the primary case and accented English is what a Chinese driver actually speaks. The 24%/39.50% pair is the answer to a different question - how the engine handles General American - and it is the reason a US-market SKU would need its own profile, not the reason the shipping path is weak.

Measured rather than assumed: running the sweep with `PF_HOTWORDS=hotwords_vehicle_replace.txt`, the profile tuned on Tingting with the Mandarin checkpoint, moved every voice down by 0.8-2.3 points en WER on that checkpoint and left zh CER bit-identical. That profile does not transfer to the English checkpoint at all - see the portability note above - so on the shipping English path its contribution is zero rather than negative. The historical value of that measurement was in showing the bias still helps off-domain and costs nothing when it does not apply; it was never the source of the remaining error.

Fixtures and scores are reproducible with:

```sh
PF_MODELS="${PF_MODELS:-$HOME/models/paraformer}"
for v in Sandy Eddy Rocko Shelley Reed Grandma Grandpa Flo; do
    python3 examples/paraformer/tools/gen_voice_fixtures.py --voice "$v" --out "/tmp/voice-$v"
done
python3 examples/paraformer/tools/compare_checkpoints.py \
    --zh "$PF_MODELS/paraformer-zh-q8_0.gguf" \
    --en "$PF_MODELS/paraformer-en-q8_0.gguf" \
    --dir /tmp/voice-Eddy --dir /tmp/voice-Rocko \
    --out /tmp/ckpt-ab.json
```

`compare_checkpoints.py` reads `voice_languages.tsv` and drops clips whose voice is not native for the clip language, so a directory listed in `--dir` may legitimately contribute zero zh clips. The English checkpoint is a conversion from the ModelScope weights; `tools/requant.cpp` produces the Q8_0 form:

```sh
build-pf-host/bin/llama-paraformer-requant "$PF_MODELS/paraformer-en-f32.gguf" \
    "$PF_MODELS/paraformer-en-q8_0.gguf" Q8_0
```

`llama-quantize` cannot do this: it accepts only llama architectures and rejects `paraformer` with `unknown model architecture`.

The fixture generator, package versions, checkpoint and clip hashes, transcript and per-clip grades, score trials and launchers are archived in `$PF_RUN/quality-funasr.7dHKEX`. Replay the upstream model with [`tools/run_funasr_reference.py`](tools/run_funasr_reference.py) and the exact dependency versions in `pip-freeze.txt`. To reproduce AISHELL-5 slicing from a local Dev archive, extract sessions 016, 017 and 018, then run the commands below with the same run directory. The corpus is publicly sourced and is kept outside the repository.

```sh
PF_OUT="$PF_RUN/quality-funasr.7dHKEX"
python3 examples/paraformer/tools/slice_aishell5.py --dev "$PF_OUT/aishell5" \
    --out "$PF_OUT/aishell5-clips" --per-speaker 6
bash "$PF_OUT/replay.sh" "$PF_OUT" reference
bash "$PF_OUT/replay.sh" "$PF_OUT" cpu
bash "$PF_OUT/replay.sh" "$PF_OUT" grade
python3 "$PF_OUT/score_aishell.py" "$PF_OUT"
```

### Long Mandarin timestamps

A 24 s fixture placed four Mandarin commands at known offsets. FunASR `fa-zh` force-aligned each source command separately. The replay shifted those intervals by their clip offsets and matched characters to the device's `verbose_json` words. Native windowing returned 48 characters, matching 39 of 40 reference characters. With Silero VAD, two spans returned exactly 40 matching characters and total inference plus VAD fell from about 2.95 to 1.80 s. In the VAD result the median absolute difference from the alignment was 82 ms for starts and 222 ms for ends. The maxima were 326 and 546 ms. Both arms emitted point timestamps for individual Mandarin characters. Treat these as CIF anchors when measuring time alignment. The model, input, references, responses and matched rows are in the quality run directory as `align-*`. [FunASR timestamp prediction example](https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/paraformer/README.md#timestamp-prediction).

## Android release bundle replay

The validated bundle used an ARM64 release build from `ghcr.io/snapdragon-toolchain/arm64-android:v0.8` with the Snapdragon CMake preset, `LLAMA_BUILD_EXAMPLES=ON` and `XASR_SPEAKER=ON`. See the [Snapdragon build guide](../../docs/backend/snapdragon/README.md#manual-cmake-build) for initializing the preset and the [speaker build](README.md#speaker-gate-optional) for configuring Sherpa headers and libraries. Build all three binaries from the established configuration.

```sh
docker run --rm --platform linux/amd64 --user "$(id -u):$(id -g)" \
    -v "$PWD:/workspace" -v /tmp/sherpa:/sherpa:ro -w /workspace \
    ghcr.io/snapdragon-toolchain/arm64-android:v0.8 cmake --build /workspace/build-android-xasr \
    --target llama-paraformer llama-paraformer-api llama-paraformer-server -j 6
```

The package groups the ARM64 executables under `bin/`, ggml and the v73 DSP library under `lib/`, Sherpa and ONNX Runtime under `onnx/bin/` and the Q8_0 weights at the bundle root. It includes the supervisor, the vehicle rewrite profile and a warmup WAV. The audit replay uses ADB to copy the package to a fresh directory, verifies every SHA-256 on-device, forwards a loopback port and starts one HTP and one CPU worker with `--prefer-idle-htp --no-partials`. Its 16 file hashes matched and all 21 API checks passed. Time from the start command through both workers warming up and returning HTTP 200 was 2.35 s. The benchmark measures a process-cold start from an already running Android device.

```sh
PF_OUT="$PF_RUN/release-sm6650.JSHel3"
bash "$PF_OUT/package.sh" "$PF_OUT"
bash "$PF_OUT/verify-package.sh" "$PF_OUT"
```

The archive includes `bundle/SHA256SUMS`, source hashes, base revision, toolchain digest, health, startup logs and the smoke checks. Optional VAD and CAM++ require their respective model files in addition to this bundle.

## Open acceptance work

- English command recognition is a speaker effect, not a language effect, and the model decision is settled. `paraformer-zh` stays as the product checkpoint because it is the only candidate that survives mixed Mandarin/English sentences, and Mandarin is the primary case. On Mandarin-accented English it reads 12.21%; `paraformer-en` would read the same clips at 4.58% but returns pinyin for the Mandarin half of a mixed sentence, so it cannot ship. What is left is that native-US-accented English costs 39.50% WER, which matters only if the product must serve American-accented speakers. Three independent engines agree on the direction, so this is audio and speaker distribution rather than a defect in this engine.
- The cabin hotword profile is now built against the mixed fixture (`tools/cabin_mixed.tsv`) and closed this round. Mixed command field 80% -> **100%**, 32-clip cabin command 56.2% -> **71.9%**, English WER 12.21% -> **6.87%**, Mandarin bit-identical, identical on host and SM6650. Retiring the old profile was the win, not adding rules: the §13 `1.5 two` gain inserted a stray `two` after `to` in `temperature to 二十一度`, so mixed scored **70%** with the old profile against 80% with none. Three rules then did the remaining work - `3.0 t@@>well` for `footal`->`footwell`, which §14 had written off as unfixable on a token-count argument that only holds for the whole word; `R 2.6 we+are+right => re@@+ar+right` for the zh->Latin boundary; and two `R` noun rewrites recovered from the deprecated file. See design doc §30-32.
- Deprecating a hotword profile is a file operation, not a rule operation. The §15 retune replaced `hotwords_vehicle_replace.txt` and dropped its two noun rewrites along with the bad `two` rule, which cost 6.3 points of cabin command until they were restored. Add this to the checklist for any future retune: diff the old and new profiles rule by rule, and move each rule that still earns its place into the new file before deleting the old one.
- Mixed-language scoring needs a command-field gate, not character CER, and `tools/eval_mixed.py` now provides it (`--run base=... --run hot=... --baseline base` fails on any field regression). It reuses `eval_vehicle.py`'s parser rather than reimplementing it, and self-checks against the reference sentences first: it must score perfect input as 100%, which is how the space-stripping and numeral defects were found.
- The previous round's SM6650 HTP failures were not device breakage. `--device HTP0:0` is resolved to `HTP0` by the device resolver, and `main()` then rejected it against the resolved name with a plain `strcmp` and returned 2 — after the session was already up and the full `ggml-hex` log had printed, which is why it read as a transport error. The check now compares device names, so the session-suffix form works and typos still fail loudly. Re-verified on SM6650: mixed command field 80% -> 100%, 32-clip cabin command 56.2% -> 65.6%, `zhCER` 0.60% unchanged, `enWER` 12.21% -> 8.40%, device transcripts identical to host including both hotword fixes. See design doc §31.
- Inject a DSP transport failure and a stuck worker to validate process isolation with the target service manager. Exercise restart under load and confirm the socket and queue resource bounds on that deployment.
- Establish release thresholds with target-vehicle command recordings from independent Mandarin and English speakers. Record overlap, speed, window state, HVAC, playback, seat and microphone for each utterance, and score the actual downstream command parser for wrong actions, extra actions and false activation. Keep voiceprint advisory until target-channel independent enrollment and impostor trials qualify it.
- Agree the user-facing P95/P99 deadlines, failure and recovery targets, soak duration and RAM budget on the production SoC with representative services active. Use the same 20 ms workload and compare the two routing modes at that load.
