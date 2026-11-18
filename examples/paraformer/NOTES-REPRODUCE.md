# Reproducing the Paraformer numbers

`./reproduce.sh` is the short answer; this note is the long one, and it exists
because one requirement cannot be met by a shell script: **the accuracy figures
are only comparable on the exact audio they were measured on**, and that audio
is rendered by a TTS engine that is not part of this repository.

Three variables cover every path a reader needs; `reproduce.sh` defaults them
and every snippet in `README.md` and `PRODUCTION.md` uses them, so there is no
machine-specific path anywhere in the documentation.

| Variable | Meaning | Default |
|---|---|---|
| `PF_MODELS` | Weights plus the fixture trees (`tingting/`, `voices/`) | `~/models/paraformer` |
| `PF_RUN` | Run directory for one evaluation or experiment | `~/pf-runs/paraformer` |
| `PF_CACHE` | Scratch/tooling root: python venv, ONNX reference exports | `~/pf-cache` |

```sh
export PF_MODELS=~/models/paraformer PF_RUN=~/pf-runs/paraformer PF_CACHE=~/pf-cache
```

## What is reproducible everywhere

| Item | How | Depends on |
|---|---|---|
| Weights | `./reproduce.sh models` -> `tools/fetch_models.py`, sha256 checked | network, ~700 MB |
| Binaries | `./reproduce.sh build` | cmake, a C++17 compiler |
| Loops / mixin logic | `tools/eval_mixed.py`, `tools/eval_vehicle.py` | python3 only |
| Latency shape | `--warmup FILE` then repeated runs | a real device for HTP |
| Relative gain of the hotword profile | `./reproduce.sh eval` on **your** fixtures | a TTS that can render 47 clips |

## What is not

The fixtures. They are 47 clips of synthetic speech: 32 cabin commands in
Mandarin and English, 15 mixed-script sentences. In the recorded runs they were
rendered with macOS `say` using the **Tingting** (zh_CN) voice, then trimmed and
converted to 16 kHz mono PCM16 by `tools/gen_voice_fixtures.py`.

`say` is a macOS component. There is no portable equivalent that produces the
same waveform: Piper, espeak-ng, Coqui and Azure all render the same sentence
differently, and Paraformer was trained on real speech, so a different waveform
gives a different error rate on identical text. That is why the committed
numbers say "Tingting fixtures" and not "the cabin set".

Two consequences, both already handled:

1. **Absolute rates do not transfer.** A reader on Linux who renders the same 32
   sentences with Piper will get a different cabin command rate - higher or
   lower, depending on the voice. That is not a regression in this repository.
2. **Relative gain mostly does.** The hotword profile was tuned on this model,
   not on this audio, so the improvement it provides survives a voice swap,
   usually with the same sign and a similar magnitude. `reproduce.sh eval` treats
   this as the gate: it compares the profile arm against the no-profile arm on
   whatever fixtures are present, and it warns explicitly when *both* arms fall
   below the recorded baseline, because that pattern means "your fixtures are
   not the recorded fixtures", not "the profile broke".

## Three ways to run the gate

**1. On macOS with `say` (exact).** This is the only path that reproduces the
committed numbers.

```sh
./reproduce.sh all
```

`fixtures` renders with Tingting via `gen_voice_fixtures.py`, and `eval` asserts
71.9% / 100.0% / 0.60% / 6.87%. If both cabin arms come in low, the render
differed - check that `say -v '?'` lists Tingting; the multilingual novelty
voices are excluded by `tools/voice_languages.tsv` for good reason (Sandy, Eddy,
Rocko, Shelley, Reed, Grandma, Grandpa and Flo render Mandarin as damaged audio
that every recogniser fails).

**2. Elsewhere, with your own fixtures (relative).** Drop `--hotwords` and see
what your audio gives, then add it back:

```sh
export PF_MODELS=~/models/paraformer
export PF_RUN=~/pf-runs/paraformer
./reproduce.sh check models build fixtures   # fixtures needs a TTS; see below
```

If no `say` is available, `gen_voice_fixtures.py` cannot run. Supply a directory
shaped like its output instead - each `.wav` plus a `manifest.tsv` of
`<name>.wav<TAB><zh|en|mixed><TAB><reference text>` - and point `PF_MODELS` at
it. Copy `tools/cabin_commands.tsv` and `tools/cabin_mixed.tsv` as the
references. Then run `./reproduce.sh eval` and read the two arms: the gate's
absolute thresholds will fail (by design - they describe Tingting), but the
base-vs-hot delta is the measurement that matters off-Mac.

**3. On the device.** `./reproduce.sh device` runs the cross-check itself:
it pushes the current 32 clips, transcribes them in three arms (host, device
CPU, device HTP0) and compares the transcript field only. Recorded on SM6650 on
2026-11-18: **32/32 identical in both device arms**. The criterion is
byte-identical transcripts - not similar latency. Latency moves with thermals
and scheduling; transcripts do not, and a transcript that differs is a real
defect.

It does not build the Android binary; that needs the docker arm64 toolchain and
a push step this script does not own. Set `DEVICE_BIN` and `DEVICE_MODEL` if the
binary and weights live somewhere other than `/data/local/tmp/llama.cpp` and
`/data/local/tmp/llama-paraformer/`.

Two ways this check can lie, both already handled:

* **Stale fixtures.** The device keeps whatever was pushed last. On the first
  run here it still held a `win_en01.wav` from an earlier round - different
  bytes, 2348 ms against the current clip's 1990 ms - and decoded `window
  upwaway` against the host's `window halfway`. That reads exactly like an HTP
  defect and is not one. The script re-pushes every run and asserts the clip
  count.
* **Blaming the wrong layer.** Device CPU runs before HTP. If CPU already
  disagrees with the host, the cause is a build or weight mismatch and the HTP
  arm is not evidence. The summary says so rather than leaving it to be guessed.

Failure path is verified as well: pointing `DEVICE_MODEL` at the en checkpoint
produces 32/32 differing, the explanation above, and exit 1.

## The one thing to remember

The fingerprint is the contract. `tools/model_manifest.txt` pins sha256 for all
four weight files, because a different conversion of the same source checkpoint
shifts the logit gaps by a few tenths, and `hotwords_vehicle.txt` was tuned
against those gaps at that precision. A verified-identical weight file
reproduces; an "equivalent" one does not, and the profile will look broken for
no visible reason.

`fetch_models.py --verify-only` runs offline and deletes anything that does not
match. `reproduce.sh check` calls it and reports a mismatch as a warning rather
than a failure - a run with unverified weights still produces a valid *relative*
measurement, it just cannot be compared to the committed numbers.