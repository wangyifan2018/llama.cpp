# Paraformer production work log

The product receives complete 16 kHz VAD segments and returns final Mandarin and English transcripts. The target device is SM6650, with up to four concurrent requests and no required partial display. The existing Q8_0 HTP + CPU configuration is the reference. Compare changes against endpoint-to-final P95 as well as failures, transcript and command-field accuracy, and device resource use.

## Request lifecycle and input boundaries

Baseline source `7d7590cdc` reproduced two WebSocket failures on macOS CPU. An empty commit did not return an error. Two consecutive accepted commits on the same connection returned only the second result.

Each admitted commit owns an item ID and returns its own terminal result or error while the connection stays open. `clear` drops uncommitted audio and invalidates previews. A full queue returns an error before acknowledging a commit and leaves buffered audio available for retry. Queued work checks for disconnect or an obsolete preview before using an inference engine. Previews only append text when the new hypothesis extends the already published prefix.

All server inputs now enforce a 30-second segment limit. TCP validates sample alignment before reading into its float buffer, caps connections at 16, and applies 10 s receive and 5 s send inactivity timeouts. HTTP bounds the complete body at 2 MiB. Raw float PCM rejects non-finite samples on both transports. WAV parsing checks the PCM format tag and complete samples, and WS validates event field types before converting them.

### Replay on the host

```sh
cmake --build build-x-asr --target llama-paraformer-api llama-paraformer-server -j 6
PF_MODELS="$HOME/Library/Caches/qwen3-asr-htp/paraformer-models"
PF_WAV="$PF_MODELS/tingting/hvac_zh01.wav"
build-x-asr/bin/llama-paraformer-api "$PF_MODELS/paraformer-zh-q8_0.gguf" \
    --bind 127.0.0.1 --port 18785 --engines 1 --threads 2 --no-partials --warmup "$PF_WAV"
```

In a second terminal, run the smoke test with Python and the `websockets` package installed.

```sh
PF_WAV="$HOME/Library/Caches/qwen3-asr-htp/paraformer-models/tingting/hvac_zh01.wav"
python3 examples/paraformer/tools/smoke_production.py --port 18785 --wav "$PF_WAV"
```

The macOS ARM64 CPU run passed all 18 checks with VAD, speaker scoring, and previews disabled. A separate loopback TCP run rejected unaligned and non-finite audio, then returned a nonempty transcript for the same WAV.

### Android verification

Cross-build both server binaries using [the Android build](README.md#openai-compatible-api) with the existing `build-android-xasr` release configuration and `XASR_SPEAKER=ON`. Run the API binary with the same options as the host replay, using device paths for the model and WAV. Add `--vad-model onnx/model/silero_vad.onnx`, include `onnx/bin` in `LD_LIBRARY_PATH`, and forward the loopback port with ADB. Pass `--expect-vad` to the smoke test.

An isolated API instance on SM6650 passed 20/20 checks with one CPU engine, two threads, VAD enabled, and previews and speaker scoring disabled. This verifies the ARM64 implementation and runs separately from the installed HTP service. The baseline observations, host and Android raw output, build logs, and source, model and binary hashes are archived in `$HOME/Library/Caches/qwen3-asr-htp/paraformer-production/lifecycle.Ojlk2S`.

## Backend readiness and inference status

The API and framed TCP service start listening only after their configured engines initialize and warm up. The framed benchmark service also retains `--no-warmup` for measuring cold starts. When CIF emits no tokens during a silence warmup, a zero embedding exercises the decoder graph. Warmup verifies that both stages execute some compute on the selected backend. `/health` keeps the engine count and reports `backends` with the requested and actual device and compute-node counts for both warmup graphs. Views and reshapes are excluded. Fbank, CIF and output conversion also run on the host and are outside these graph totals.

An unavailable requested accelerator or failed warmup exits before listening by default. `--allow-cpu-fallback` opts into a CPU startup retry. The API reports HTTP 200 and `status: degraded` in that case, and `/metrics` reports `cpu_fallback: true`. A valid no-token decode returns empty text successfully. Failed graph allocation or compute returns an inference error through plain, timestamped, and windowed decoding. The graph cache drops failed stages instead of reusing them. The temporary source GGUF bytes are released after the backend buffers are materialized.

The host run verified strict failure with missing HTP, strict failure with a sub-frame warmup WAV, and successful CPU fallback with `degraded` and `requested=HTP0:0 actual=CPU`. Both ordinary CPU and fallback CPU passed 20 smoke checks. With the same WAV in a separate SM6650 instance, HTP performed 1454/1504 encoder and 559/575 decoder compute nodes. All 66 nodes outside HTP were `FLASH_ATTN_EXT`. Both engines warmed up and the mixed instance passed 20 smoke checks, including empty text from a valid sub-frame PCM request. This is a placement and API check at the warmup shape. Compare attention paths within the same API binary before changing the serving default.

To replay, use the host and Android builds and model above. On macOS run with `--device HTP0:0` and expect startup exit 3, then add `--allow-cpu-fallback` and run the smoke script with `--expect-backend CPU --expect-degraded`. On SM6650 run the HTP launch from the [API section](README.md#openai-compatible-api) with `--engines 2 --threads 2 --no-partials` and a real speech warmup WAV. Pass `--expect-backend HTP0:0` to the smoke script. The exact launchers, source and device hashes, and raw results are in `$HOME/Library/Caches/qwen3-asr-htp/paraformer-production/readiness.hzWCkj`.

The Hexagon backend can abort the process on a DSP error during synchronization or tensor readback. Runtime fault isolation and retry still need a recovery path that can survive termination of the accelerator worker. The startup fallback above covers initialization and recoverable warmup failures.

## Hotword rewrite timestamps

Each decoder choice keeps its source CIF position. A length-changing rewrite keeps both endpoints of the original span. Its interior positions interpolate between the original CIF rows. Choices after the rewrite retain their original rows for both scoring and timestamps. Rewritten tokens use the acoustic probability of the replacement at the nearest original row. These timings describe CIF anchors, so precise acoustic word boundaries still require alignment against the reference. CLI token times, the centisecond C API, and REST word spans derive from the same mapped records. Windowed decoding adds the global offset to both ends of each span.

The Q8_0 shipping profile changed `likes` from two to three pieces within 720-1020 ms and `footwall` from five to four pieces within 300-720 ms, on macOS CPU and SM6650 HTP. The unchanged suffixes kept their original timestamps. The final `light` of `light_en03.wav` stays at 2340 ms, where it previously flipped to 0 ms. Plain and timed CLI transcripts agreed on all 32 Tingting fixtures. A 22 s fixture with six commands and silence exercised the model's windowed path. On the SM6650 REST API with server VAD off, 52 ordered word spans ended at 20.060 s.

```sh
PF_MODELS="$HOME/Library/Caches/qwen3-asr-htp/paraformer-models"
PF_WAVS="$PF_MODELS/tingting"
PF_MODEL="$PF_MODELS/paraformer-zh-q8_0.gguf"
PF_PROFILE=examples/paraformer/hotwords_vehicle_replace.txt
build-x-asr/bin/llama-paraformer "$PF_MODEL" "$PF_WAVS/light_en03.wav" "$PF_WAVS/light_en04.wav" \
    --threads 2 --timestamps > rewrite-off.tsv
build-x-asr/bin/llama-paraformer "$PF_MODEL" "$PF_WAVS/light_en03.wav" "$PF_WAVS/light_en04.wav" \
    --threads 2 --timestamps --hotwords "$PF_PROFILE" > rewrite-on.tsv
python3 examples/paraformer/tools/check_rewrite_times.py rewrite-off.tsv rewrite-on.tsv
```

The run archive holds the long fixture generator, isolated Android replay and REST assertions, both CLI TSV arms, REST JSON, and artifact hashes under `$HOME/Library/Caches/qwen3-asr-htp/paraformer-production/rewrite-times.AkZMgI`.

## Final request VAD and long segments

WAV, raw PCM and WebSocket final requests now share a single admission and decode path. Digital silence returns without model inference even with server VAD off. When enabled, each engine owns one validated VAD detector and resets it before each analysis. Ordinary short speech only pays the RMS check. Long clips group detected spans, retain up to 100 ms of nonoverlapping edge context and send each actual slice to the model, including a single detected slice. WAV verbose responses retain words for every slice and offset them to the original audio. Silence responses preserve the requested format. Each admitted segment occupies one worker for all its slices. The impact of long requests on short-command queueing remains part of the scheduling tests. The framed TCP benchmark server uses the model's native windows.

On SM6650, one HTP engine with Silero VAD passed 22 smoke checks. WAV, raw PCM and WS matched on five fixture replays. The 3 s command stayed one slice. A 22 s clip with a single late command returned one cropped slice and 11 words spanning 15.084-17.484 s in the original input. A 22 s clip with several commands returned two slices and 54 words spanning 0.076-20.052 s. Repeating the late command after the multi-slice clip gave identical word times. The long-clip VAD step took about 140 ms in the first replay. The same replay measured under 0.1 ms for the short command's energy check. macOS CPU with server VAD off passed 20 smoke checks, including digital silence.

Run the Android API with `--vad-model onnx/model/silero_vad.onnx --engines 1 --threads 2 --no-partials`, the Q8_0 model and the vehicle rewrite profile. Forward the loopback port and run the Python verifier with `websockets` installed.

```sh
PF_WAVS="$HOME/Library/Caches/qwen3-asr-htp/paraformer-models/tingting"
python3 examples/paraformer/tools/check_vad_ingress.py --port 18788 \
    --wav "$PF_WAVS/light_en03.wav" "$PF_WAVS/light_en04.wav" --out /tmp/pf-vad-replay
```

The build log, device hashes, launcher, fixture generators and per-transport outputs are archived in `$HOME/Library/Caches/qwen3-asr-htp/paraformer-production/vad-ingress.ARo9fu`.

## Bounded speaker scoring

The API runs CAM++ scoring on one CPU thread and admits at most one scorer task, including its active task. It tries to schedule scoring only when a final request is admitted, the ASR queue and workers are idle, and the scorer slot is free. It copies at most the first 1.5 s by default. Final replies inspect the result once at ASR completion. A slower scorer can finish independently as the ASR result is delivered. The requester does not receive a later update for that result.

Both REST endpoints and WS final responses distinguish `disabled`, `assessed`, and `not_assessed`. `speaker_ok` exists only for `assessed`. The not-assessed reasons are `busy`, `in_progress`, `no_audio_in_prefix`, `no_embedding`, and `silence`. `/metrics` reports scorer starts and final assessment counts. Supplying a speaker configuration with unavailable model support or enrollment fails before listen. The raw PCM endpoint follows the same policy as WAV and WS.

On SM6650 with the same binary on both arms, six rounds of four aligned WS requests completed with identical transcripts in 24/24 pairs. Enabled scoring started once per round. Six responses carried a score and 18 returned `not_assessed` with `busy`. Endpoint-to-final P50/P95 was 527/724 ms with scoring off and 547/729 ms with scoring on. This was one short off/on comparison. The enabled configuration passed 26 smoke checks. A sub-frame request returned `not_assessed` with `in_progress`, and WS silence returned `not_assessed` with `silence`. Synthetic enrollment and independent synthetic speech yielded example scores of 0.926 and 0.339. These samples establish response semantics and exercise the scorer. Cabin false accept and false reject rates are still open.

Build Android with `XASR_SPEAKER=ON` and launch [the API](README.md#speaker-gate-optional) with two engines (`HTP0:0` + CPU), two inference threads, previews off and server VAD off. Replay with the same 2.32 s mono fixture and no speaker flags for the off arm. Enable `--speaker-model` and `--enroll-dir` for the on arm. Install `websockets` for both commands.

```sh
PF_WAV="$HOME/Library/Caches/qwen3-asr-htp/paraformer-models/tingting/hvac_zh01.wav"
python3 examples/paraformer/tools/smoke_production.py --port 18789 --wav "$PF_WAV" --expect-backend HTP0:0 --expect-gate
python3 examples/paraformer/tools/bench_ws_realtime.py --url ws://127.0.0.1:18789/v1/realtime \
    --metrics http://127.0.0.1:18789/metrics --wav "$PF_WAV" --sessions 4 --rounds 6 --out speaker-on.jsonl
```

The Android launcher, synthetic enrollment fixture identifiers and hashes, raw response JSON, per-commit JSONL, and counters before and after each arm are in `$HOME/Library/Caches/qwen3-asr-htp/paraformer-production/speaker.5xc2A3`.

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
PF_CACHE="$HOME/Library/Caches/qwen3-asr-htp"
PF_WAVS="$PF_CACHE/paraformer-models/tingting"
python3 examples/paraformer/tools/bench_ws_realtime.py --url ws://127.0.0.1:18791/v1/realtime \
    --metrics http://127.0.0.1:18791/metrics --sessions 4 --rounds 2 \
    --wavs "$PF_CACHE/cabin-asr-probe/long-perf/zh-10.wav" "$PF_WAVS/hvac_zh01.wav" \
    "$PF_WAVS/light_en01.wav" "$PF_WAVS/light_en03.wav" --start-offsets-ms 0 7700 8050 7570 \
    --out mixed.jsonl
```

The exact launcher, quality replay, raw requests, backend placement, hashes, memory and thermal snapshots are archived in `$HOME/Library/Caches/qwen3-asr-htp/paraformer-production/attention-ab.rPnRkA`.

## HTP encoder width-two probe

`llama-paraformer --bench-encoder-pair REPS` compares two width-one encoder executions with one width-two execution. Both arms use the same weights, fbank features and sinusoidal positions. Width two flattens the time and batch axes only for the shared-weight projections and FFN. Attention and FSMN retain a separate axis for each clip. The probe warms both graphs twice, alternates execution order and swaps the clips across lanes. Wall time covers input upload, execution and output readback. Features, graph allocation, the CIF predictor and the decoder are outside the timed region.

The SM6650 run used Q8_0, decomposed attention, two threads and two distinct clips per width. It dispatched all 1704 compute nodes to HTP in one scheduler split at both widths and lengths. All encoder output values matched exactly, including the swapped lanes. The macOS CPU control also matched exactly. With the vehicle rewrite profile applied, the ordinary width-one CPU CLI returned the same tokens, text and timestamps as the saved baseline on 32/32 clips.

| Equal audio length per lane | LFR frames | Two width-one executions P50 | One width-two execution P50 | Pair time saved |
|---|---:|---:|---:|---:|
| 2.70 s | 45 | 483.24 ms | 470.27 ms | 12.97 ms (2.7%) |
| 5.40 s | 90 | 944.05 ms | 929.22 ms | 14.83 ms (1.6%) |

Keep this as a CLI benchmark. The pair saves about 13-15 ms when both rows are already available. The first item in a serial pair can finish its encoder stage much earlier than the batched pair, which matters for staggered commits. Service scheduling continues to submit one segment per engine call.

```sh
PF_CACHE="$HOME/Library/Caches/qwen3-asr-htp"
PF_OUT="$PF_CACHE/paraformer-production/encoder-b2.IENc7B"
python3 "$PF_OUT/make_fixtures.py" "$PF_CACHE/paraformer-models/tingting" "$PF_OUT"
build-x-asr/bin/llama-paraformer "$PF_CACHE/paraformer-models/paraformer-zh-q8_0.gguf" \
    "$PF_OUT/short-a.wav" "$PF_OUT/short-b.wav" --threads 2 --bench-encoder-pair 6
bash "$PF_OUT/android.sh" "$PF_OUT"
```

The fixture generator zero-pads `hvac_zh01.wav` and `light_en03.wav` to 43,200 samples and doubles each padded sequence for the long arm. The run directory holds input and binary hashes and raw logs from six repetitions per host shape and ten repetitions per device shape.

## HTTP and WebSocket admission and deadlines

The API caps its connection pool at 12 active socket handlers with up to eight sockets queued. Four WebSocket sessions can be active. A fifth upgrade is closed with code 1008. Each session admits up to four pending commits. `--queue` and `--queue-mib` bound ASR requests waiting for an engine by count and allocated PCM capacity, with defaults of 64 and 8 MiB. A rejected WS commit retains the audio for retry. Explicit `clear` releases the buffer allocation. HTTP returns 429 when a request exceeds the admission budget and 400 for an invalid segment.

`--deadline-ms` defaults to 30,000 ms and starts at final request submission, after HTTP body intake or WS COMMIT. REST returns 504 with `deadline_exceeded`. A WS commit keeps its item ID and receives one terminal `error` with the same code. The WS poll loop checks for expired items every 40 ms when idle. The worker skips expired queued work and drops the result of a completed decode if it missed the deadline. A model call already running continues to hold its engine until it returns. `--timeout-ms` continues to cap the wait for an engine and returns 503 when that limit is reached first.

```sh
PF_MODELS="$HOME/Library/Caches/qwen3-asr-htp/paraformer-models"
PF_WAV="$PF_MODELS/tingting/hvac_zh01.wav"
build-x-asr/bin/llama-paraformer-api "$PF_MODELS/paraformer-zh-q8_0.gguf" \
    --bind 127.0.0.1 --port 18797 --engines 1 --threads 2 --no-partials --warmup "$PF_WAV" \
    --queue-mib 1 --deadline-ms 20
# From a second terminal with websockets installed, or against the forwarded device port
python3 examples/paraformer/tools/check_admission.py --port 18797 --wav "$PF_WAV"
```

The host CPU and SM6650 HTP both rejected an over-budget PCM request, rejected a fifth session, let a WS client retry a rejected commit, returned three deadline terminal responses across WAV, raw PCM and WS, suppressed a late duplicate result, and accepted a reconnect. Each run ended with zero queued bytes and zero in-flight requests. The 20 ms profile is a test configuration. Launchers, hashes and raw responses are in `$HOME/Library/Caches/qwen3-asr-htp/paraformer-production/admission.iALkbe`.

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

The isolated build and launch commands, metrics, per-request JSONL, parts of `/proc`, thermal outputs and artifact hashes are in `$HOME/Library/Caches/qwen3-asr-htp/paraformer-production/admission.iALkbe`. Re-run with `bash "$PF_OUT/device.sh" "$PF_OUT"` from the repo, with `PF_OUT` set to that run directory. Device scripts use temporary paths and an ADB loopback forward.

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

The ABBA launchers, per-request results with `backend`, health, artifact hashes, and resource snapshots are in `$HOME/Library/Caches/qwen3-asr-htp/paraformer-production/routing.vjddv2`. Use `bash "$PF_OUT/route.sh" "$PF_OUT"` and `bash "$PF_OUT/pressure.sh" "$PF_OUT"` from the repo with that run directory as `PF_OUT`. The recorded pressure hash is from the phone. The initial routing executable was reconstructed from the archived source version after the manifest file was reused for the pressure run. It matches the logged transfer size. The model and backend libraries were identical in both runs.

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

On SM6650, sending SIGKILL to the isolated HTP child after an acknowledged WS COMMIT closed that connection. Health returned `degraded` with two CPU engines on the same port after 1.57 s. The client opened a new socket, resent audio and received a correctly timed CPU result. The recovered child passed 21/21 smoke checks. The injection tests process death, the supervisor transition and client recovery. DSP faults and hangs require separate injection. A process that hangs needs the service manager's watchdog. Recovering an interrupted WS commit requires the client to retain and resubmit its audio after reconnect. The hashes, injection script, event output and supervisor log are in `$HOME/Library/Caches/qwen3-asr-htp/paraformer-production/admission.iALkbe`.

## Cabin command accuracy and FunASR

The 96-clip regression uses 32 command texts across window, climate, seat and cabin light functions, synthesized in two voice sets. A further 16 confusable clips include four commands and 12 controls. Another 16 clips mention cabin devices without issuing a command. The shared offline evaluator re-parsed all 96 annotations before scoring. `device` includes the required position and `value` includes the annotated level, number, percentage or color. The command metric requires every annotated clause to match. Potential false activation is output that the offline evaluator maps to an actionable clause. Ship decisions require an evaluation of the actual downstream command parser and extra actions as well.

The upstream CPU reference used the plain FunASR Paraformer checkpoint with VAD and punctuation off. The phone run used the Q8_0 API with vehicle rewrites and 600 ms acoustic punctuation. Keep this postprocessing difference in view when reading command counts. The host comparison includes raw and postprocessed runs of the local engine. The source checkpoint is [FunASR Paraformer on ModelScope](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch).

| Arm | Tingting command | Second voice command | Tingting zh CER / en WER | Second voice zh CER / en WER | False triggers in 28 controls |
|---|---:|---:|---:|---:|---:|
| FunASR CPU, raw | 17/32 | 6/32 | 0.60% / 12.98% | 30.12% / 21.37% | 0/28 |
| Q8_0 CPU, with punctuation | 21/32 | 8/32 | 0.60% / 12.21% | 34.34% / 22.90% | 0/28 |
| Q8_0 CPU, rewrites and punctuation | 26/32 | 9/32 | 0.60% / 6.87% | 34.34% / 18.32% | 0/28 |
| Q8_0 SM6650 HTP, rewrites and punctuation | 26/32 | 9/32 | 0.60% / 7.63% | 34.94% / 19.08% | 0/28 |

On HTP the first and second voice sets got 37/42 and 20/42 device-position clauses, and 18/18 and 5/18 value clauses. Negation and exclusion cases scored 3/4, 1/4 and 2/3 across the two voice sets and the confusable pack. The second voice exposes a release blocker even for the upstream model. Keep the existing profile as a regression arm and avoid deriving new substitutions from the holdout run.

Real Mandarin replay used 36 non-overlapping conversational clips from six speakers across three AISHELL-5 Dev sessions (DX01C01, 100 ms padding). The transcript has 449 reference characters. On SM6650 the Q8_0 API made 106 errors (23.61% CER) versus 104 (23.16%) for FunASR CPU. The offline grader saw zero modeled command triggers in those 36 clips for either arm. This small sample measures far-field conversational speech and provides negative controls. It contains no annotated bilingual positive vehicle command set. [AISHELL-5 dataset and license](https://www.openslr.org/159/).

The same six-speaker sample exercised the actual Android CAM++ path, with two disjoint enrollment clips per profile and four test clips per speaker. All 144 comparisons returned scores. At the shipping example threshold of 0.55, 5/24 genuine trials rejected (20.8%) and 30/120 impostor trials accepted (25%). At 0.65 those counts were 8/24 and 12/120. Results from shared sessions and clips are correlated and the cohort is small. The scores support retaining speaker assessment as advisory data while the control path is unqualified.

The fixture generator, package versions, checkpoint and clip hashes, transcript and per-clip grades, score trials and launchers are archived in `$HOME/Library/Caches/qwen3-asr-htp/paraformer-production/quality-funasr.7dHKEX`. Replay the upstream model with [`tools/run_funasr_reference.py`](tools/run_funasr_reference.py) and the exact dependency versions in `pip-freeze.txt`. To reproduce AISHELL-5 slicing from a local Dev archive, extract sessions 016, 017 and 018, then run the commands below with the same run directory. The corpus is publicly sourced and is kept outside the repository.

```sh
PF_OUT="$HOME/Library/Caches/qwen3-asr-htp/paraformer-production/quality-funasr.7dHKEX"
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
PF_OUT="$HOME/Library/Caches/qwen3-asr-htp/paraformer-production/release-sm6650.JSHel3"
bash "$PF_OUT/package.sh" "$PF_OUT"
bash "$PF_OUT/verify-package.sh" "$PF_OUT"
```

The archive includes `bundle/SHA256SUMS`, source hashes, base revision, toolchain digest, health, startup logs and the smoke checks. Optional VAD and CAM++ require their respective model files in addition to this bundle.

## Open acceptance work

- Inject a DSP transport failure and a stuck worker to validate process isolation with the target service manager. Exercise restart under load and confirm the socket and queue resource bounds on that deployment.
- Establish release thresholds with target-vehicle command recordings from independent Mandarin and English speakers. Record overlap, speed, window state, HVAC, playback, seat and microphone for each utterance, and score the actual downstream command parser for wrong actions, extra actions and false activation. Keep voiceprint advisory until target-channel independent enrollment and impostor trials qualify it.
- Agree the user-facing P95/P99 deadlines, failure and recovery targets, soak duration and RAM budget on the production SoC with representative services active. Use the same 20 ms workload and compare the two routing modes at that load.
