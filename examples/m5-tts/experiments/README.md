# Experiment archive: the A/B scripts behind the optimization log

Every number in
[`docs/backend/snapdragon/m5-tts-optimization-log.md`](../../../docs/backend/snapdragon/m5-tts-optimization-log.md)
came from one of the scripts in this directory. They are archived here on
purpose: a conclusion without its harness is a rumour, and the harness is the
only thing that lets a reader re-derive — or refute — the claim on a different
device.

They are **evidence, not product code**. Each script hard-codes the fixed
sentence, the seed and the pass/fail criterion that the corresponding log
section used; most of them print raw `key | json` lines instead of judging. Read
the section, then read the script.

## Layout

```
experiments/
  *.sh, *.py, *.mjs         the harnesses (27 shell + 2 python + 1 node)
  results/*.txt, *.log      raw output captured on SM6650 (12 + 7)
  audio/*.wav               two listening samples for the chunking decision
  voices-web.json           voice library used by prime-lru.sh (65x16 codes)
```

## How to run one

Two flavours, marked "device" / "host" in the table below:

- **device** — `#!/system/bin/sh`, pushed to the phone and run there next to
  `m5-server` / `m5-engine`; the stage processes are children of the shell.
- **host** — `bash`/`python`/`node` on the desktop, driving the device through
  `adb` or the forwarded HTTP port `127.0.0.1:18080`.

```sh
# host-side example (server must already be running, see deploy.sh)
adb forward tcp:18080 tcp:8080
bash experiments/wavchunk-ab.sh

# device-side example
adb push experiments/kv32-ab.sh /data/local/tmp/llama-m5-htp-v73/
adb shell 'cd /data/local/tmp/llama-m5-htp-v73 && sh kv32-ab.sh'
```

Three things are parameterised so the scripts survive a rename; everything else
is byte-identical to what produced `results/`:

| variable | default | meaning |
|---|---|---|
| `ADB` | `adb` | path to the adb binary |
| `M5_ASSETS` | `$HOME/m5-assets` | where result files are written |
| `M5_VOICES` | `./voices-web.json` | voice library for `prime-lru.sh` |

The device working directory `/data/local/tmp/llama-m5-htp-v73` is still
hard-coded (it is what `deploy.sh` creates); edit the first `cd`/`V=` line if
you deploy elsewhere.

Always run the full alternating sequence a script encodes. Every one of them
interleaves forward and reverse rounds on purpose — §18 of the log explains why
a single-shot A/B on this phone is worthless (thermal drift, `max()` coupling,
and RTF moving in the opposite direction when output length changes).

## Index

`§` points at the section of the optimization log that states the result.

### Baseline, equivalence and the main knobs

| script | where | what it does | verdict | § |
|---|---|---|---|---|
| `kv32-ab.sh` | device | KV pad 32 (new default) vs 256 (upstream) vs no direct-V, 69-frame sentence | **adopted**: 3.72 vs 4.15 s, codes/WAV byte-identical | §16 |
| `bbkv-ab.sh` | device | backbone-only KV pad 256/128/64/32, injected per-subprocess | **adopted**: pad 128, wall −2.3% | §21.6, §22.1 |
| `bbkv-long-ab.sh` | device | same A/B on a 170-frame sentence, where graphs get rebuilt | pad 128 still wins (8.313 vs 8.375 s) | §22.1 |
| `p0-ab.sh` | device | turn depth/codec overlap off, then re-sweep threads and chunk size by end-to-end wall | overlap is worth ~2.4 s; thread/chunk already optimal | §13 |
| `p0-affinity.sh` | device | `taskset` the whole process tree (little cores / big cores / cpu7 @80%) | **negative**: binding is 2–3x slower | §13 |
| `oppoll-cpu.sh` | device | `GGML_HEXAGON_OPPOLL=1` (busy poll) vs `0` (blocking) | busy poll is the cheaper of the two | §13 |
| `p0-flash-ab.sh` | device | fused attention `flash-f32` vs ordinary attention at FP32 KV | **negative**: not equivalent, no end-to-end gain | §17 |
| `bbquant-ab.sh` | device | backbone intermediate quantisation Q6_K / Q5_K_M / mixed | **negative**: −48% size buys −3% time | §23 |

### First packet

| script | where | what it does | verdict | § |
|---|---|---|---|---|
| `firstpcm-ab.sh` | device | splits first-packet latency with existing switches only, `--verbose` step-0 stamps | fit: fixed 0.190 s + 74.6 ms/frame | §19 |
| `firstpcm-server.sh` | host | same on the resident server (wav headers, pcm via `/metrics`) incl. voice switch | cold 0.507 → hit 0.426 s | §19, §20 |
| `firsttail-ab.sh` | device | first-chunk tail alignment: `--first 2` default vs `--first 1` with `M5_FIRST_TAIL=0` | **negative**: `--first 1` is slower because of the tail keep | §22.2 |
| `prefix-xfer-ab.sh` | host | how much of the prefill round trip is the 32-row shared prefix on a cache hit | **negative**: not worth a protocol change | §22.3 |

### Caches

| script | where | what it does | verdict | § |
|---|---|---|---|---|
| `prime-ab.sh` | host | codec prime cache capacity 0 / 1 / 4, alternating, serena↔official | **adopted**: cap 4, hit 0.500 → 0.463 s | §20 |
| `prime-lru.sh` | host | six mutated code sets in an order that forces LRU eviction at cap 4 | content-keyed LRU verified | §20 |

### Concurrency and CPU accounting

| script | where | what it does | verdict | § |
|---|---|---|---|---|
| `cpuprof.sh` | device | sums utime+stime of the engine and both stage children per request | 10.62 CPU-s per request, ≈4.4 cores | §26 |
| `cpustat.sh` | device | whole-device `/proc/stat` average busy cores in the request window | cross-check of the above | §26 |
| `srvcpu.sh` | device | CPU accounting on the resident server, excluding ORT session startup | isolates the CLI startup tax | §26 |
| `srvcpu2.sh` | device | v2: `wait` only on curl, samples 8-core clocks and governor | explains the run-to-run jitter | §26 |
| `multi-inst.sh` | device | N `m5-engine` instances on the same sentence | `wall(N)/wall(1) ≈ 1.87` → time-sharing, not parallel | §29 |
| `dual-srv.sh` | device | two servers (8080/8081), each with its own stage processes | still sharing one HTP0 | §29 |
| `conc-ab.sh` | host | concurrency 1/2/4 (wav) and 1/2/4/8 (pcm) TTFB + batch wall | +6.6% throughput for ×1.87 latency | §26, §29 |
| `conc-pcm.py`, `conc-pcm2.py` | host | pcm streaming variant; v2 fixes the header-based first-packet measurement | same conclusion, chunked path | §26 |

### Environment, product path, codec backend

| script | where | what it does | verdict | § |
|---|---|---|---|---|
| `thermal-baseline.sh` | host | resident server back to back, NPU/DSP/CPU/DDR thermal zones + clocks | no drift; the historical +21% is not reproducible | §24 |
| `cli-soak.sh` | device | 12 consecutive CLI runs, i.e. a fresh HTP session each time | the drift was the CLI startup tax | §24 |
| `wavchunk-ab.sh` | host | sweeps `(first, chunk)` on the real wav product path | **adopted**: (2,4) over the old (4,16); −10% wall, −34% first packet | §25 |
| `ep-ab.sh` | host | codec execution provider: ORT CPU vs XNNPACK | **negative**: no gain | §28 |
| `web-e2e.mjs` | host | headless Chrome drives `web.html` end to end (synthesis, playback, stats) | regression harness for the console | §11 |
| `emo-run.sh`, `emo-seed.sh` | device | `--caption` emotion tags and `--seed` sensitivity | exploratory; establishes that seed changes frame count and codes, so A/B must pin it | — |

## Results

| file | produced by |
|---|---|
| `results/kv32-results.txt` | `kv32-ab.sh` |
| `results/p0-results.txt` | `p0-ab.sh` |
| `results/p0-aff-results.txt` | `p0-affinity.sh` |
| `results/p0flash-results.txt` | `p0-flash-ab.sh` |
| `results/firstpcm-results.txt` | `firstpcm-ab.sh` |
| `results/firstpcm-server-results.txt` | `firstpcm-server.sh` |
| `results/prefix-xfer-results.txt` | `prefix-xfer-ab.sh` |
| `results/prime-ab-results.txt`, `results/prime-ab-logs/` | `prime-ab.sh` |
| `results/prime-lru-results.txt`, `results/prime-lru.log` | `prime-lru.sh` |
| `results/conc-results.txt` | `conc-ab.sh` |
| `results/thermal-results.txt` | `thermal-baseline.sh` |
| `results/wavchunk-results.txt` | `wavchunk-ab.sh` |

The `p0*` files are the 94-frame fixed sentence (`p0` = the §13/§17 series);
the `kv32`/`firstpcm`/`p0flash` files are the 69-frame navigation sentence.
`H=` in a result line is the codes/WAV hash pair used as the equivalence oracle.

## audio/

Two renders of the same sentence and voice, differing only in
`(M5_WAV_FIRST, M5_WAV_CHUNK)`:

| file | chunking | sha256 (first 8) | why it is here |
|---|---|---|---|
| `listen-new-2-4.wav` | 2 / 4 (new default) | `2f30728f` | the shipped configuration |
| `listen-old-4-16.wav` | 4 / 16 (old default) | `405830fa` | the configuration it replaced |

Both are 24 kHz mono PCM16, 5.32 s, 255,404 bytes. The waveform difference is
1,379 of 127,680 samples (1.08%), all inside 5.032–5.090 s — one codec
overlap-add window, max delta 1,299/32,767 (~4%). ASR transcripts are
identical; this pair is what the human listening check (§25.3) was done on.
`(2,4)` is byte-identical to the CLI path, which is why it became the default.

## What is deliberately not here

Model weights (1.3 GB ONNX package), quantisation intermediates, HTP profile
dumps (up to 42 MB each), build trees and the ORT AAR. All of them are either
re-downloadable or re-buildable; see
[`../MODEL_PACKAGE.md`](../MODEL_PACKAGE.md) for the artifact table and
[`../reproduce.sh`](../reproduce.sh) for the commands that regenerate them.
