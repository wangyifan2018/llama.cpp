# Cabin command quality baseline

Generate the dedicated synthetic set and run the existing CPU evaluator on
macOS. Change the TTS voice with `ZH_VOICE` or `EN_VOICE` if needed and
capture the generated WAV hashes. Record the voices and host version with
any result that will be used for a comparison.

```sh
XASR_EVAL_TEXTS=examples/x-asr/tools/vehicle_commands.tsv \
    sh examples/x-asr/tools/make_eval_pack.sh /tmp/xasr-vehicle
python3 examples/x-asr/tools/eval_cer.py \
    --manifest /tmp/xasr-vehicle/manifest.tsv --model x-asr-zh-en-q8_0.gguf \
    --bin build-x-asr/bin/llama-x-asr --label vehicle-host-cpu
```

This set contains 32 commands, eight each for windows, climate, seats, and
interior lights, split evenly across Mandarin and English. The evaluator
reports CER/WER. Grade control semantics independently by action, target,
value and polarity before making any vehicle-command accuracy claim.

## Host CPU baseline

The same 32 references were synthesized with two Mandarin voices. Both used
Samantha for English. These values describe these WAVs and this build.

| Mandarin voice | zh CER | en WER | clips with errors |
|---|---:|---:|---:|
| Flo, Mainland Mandarin | 55.42% (92/166) | 5.34% (7/131) | 21/32 |
| Tingting | 8.43% (14/166) | 5.34% (7/131) | 15/32 |

The voice choice materially changes the Mandarin baseline. Tingting's run
still contains `氛围灯 → 分为灯` and `座椅 → 座以/座已`, so these are concrete
phrase-bias candidates. Also assess location omissions, numeric values and
negation against real segments. A successful change should improve the
intended command without turning an unrelated utterance into a command.

Waveforms, reference manifests, and hashes are in
`$HOME/Library/Caches/qwen3-asr-htp/xasr-multistream/vehicle-baseline.QMMIN2`.
Tingting's artifacts are in its `tingting` subdirectory. Set `ZH_VOICE=Tingting`
before the generator command to reproduce that voice.

## Grading by field instead of by character error

CER is the wrong headline for a command set. `Set the passenger seat heater to
level two` recognised as `said the passenger seat peter to level two` costs six
characters out of ninety-five, and no seat heater is addressed at all. The
evaluator therefore parses each transcript into clauses of device, position,
on/off state and value, and compares them with `tools/vehicle_semantics.tsv`.

```sh
python3 examples/x-asr/tools/eval_vehicle.py --check \
    --pack tingting="$PACK/tingting/manifest.tsv" \
    --pack holdout="$PACK/holdout/manifest.tsv" \
    --pack controls="$PACK/controls/manifest.tsv" \
    --pack hardneg="$PACK/hardneg/manifest.tsv"
```

`--check` re-parses every reference sentence and fails unless the fields it
extracts equal the annotation. That gate is the reason a parser bug cannot be
mistaken for an ASR error: the instrument is verified against ground truth
before it grades anything. All 96 references agree.

The annotation is `device|position|state|value` per clause. `fl`/`fr` are the
front seats, so 主驾 and `driver` mean `fl` and 副驾 and `passenger` mean `fr`.
Two clauses carry `?` for state because their own reference drops the verb
(`Only the rear right window.`); those are graded on device and position only.

Counters, in increasing severity:

| counter | meaning |
|---|---|
| device | the device and its position were recognised |
| state | the on/off direction was recognised |
| value | the level, percentage, temperature or colour was recognised |
| clause | all three of one clause |
| command | every clause of the utterance |
| wrong direction | device recognised with the opposite state, so acting on it does the opposite of what was asked |
| unstated | device and value recognised but no verb, so nothing would execute |
| false trigger | a clip annotated `none` produced a device with a stated state |
| near miss | a `none` clip produced a device word without a direction |

`none` covers ordinary speech, homophone traps and a new hard-negative pack in
`tools/vehicle_hard_negatives.tsv`: statements that name a controllable device
without commanding it (`顶灯坏了，一直在闪`, `The reading light is too bright`,
`My seat heater died last winter`). Their purpose is the false-trigger rate,
not coverage of a device.

## Optional device-phrase bias

Pass `--hotwords examples/x-asr/tools/vehicle_hotwords.txt` to the CLI or
server to enable the cabin phrase profile. Copy the file to the phone before
using the option on Android. The profile covers windows, climate, seats, and
interior lighting in both languages. It boosts device phrases during greedy
token selection. The original joiner decision gates each frame before the
boosted decision can take effect. In particular, an original blank, unknown
token, or end marker stays unchanged for the current predictor state.

The file has one phrase per line and uses a gain followed by whitespace-separated
pieces from the GGUF vocabulary. A line beginning with `#` is a comment.
Each entry uses 2–16 tokens and a gain in (0, 3]. The loader caps the profile
at 128 entries. The first piece receives 30% of the configured gain and
continuations receive the full gain when prior emitted tokens match the
prefix. `XASR_HOTWORD_HEAD=<fraction>` overrides that 30% for measurement; the
default is unchanged. Bias vectors are maintained per stream. The built-in profile gives
device words a gain of 2 and leaves action, position, negation and value
tokens out of the list. A changed device token can still affect later decoder
decisions, so always test the full transcript.

The sample pieces were generated using the released `bpe.model` and
SentencePiece 0.2.3. For Mandarin, separate Han characters by spaces before
encoding, for example `座 椅` produces `▁座 ▁椅`. Verify every produced
piece against the shipped tokens before using a custom profile.

### Reproduce the A/B

Use the same files for each arm. Keep the original Tingting/Samantha clips
for the tuning replay, then prepare a holdout and negative/polarity controls.

```sh
HOTWORDS=examples/x-asr/tools/vehicle_hotwords.txt
ZH_VOICE='Eddy (中文（中国大陆）)' XASR_EVAL_TEXTS=examples/x-asr/tools/vehicle_commands.tsv \
    sh examples/x-asr/tools/make_eval_pack.sh /tmp/xasr-hotword-holdout
ZH_VOICE=Tingting XASR_EVAL_TEXTS=examples/x-asr/tools/vehicle_hotword_controls.tsv \
    sh examples/x-asr/tools/make_eval_pack.sh /tmp/xasr-hotword-controls
ZH_VOICE=Tingting XASR_EVAL_TEXTS=examples/x-asr/tools/vehicle_hard_negatives.tsv \
    sh examples/x-asr/tools/make_eval_pack.sh /tmp/xasr-hotword-hardneg

build-x-asr/bin/llama-x-asr x-asr-zh-en-q8_0.gguf CPU /tmp/xasr-hotword-controls/*.wav > baseline.tsv
build-x-asr/bin/llama-x-asr x-asr-zh-en-q8_0.gguf CPU /tmp/xasr-hotword-controls/*.wav \
    --hotwords "$HOTWORDS" > biased.tsv
for run in baseline biased; do
    python3 examples/x-asr/tools/eval_cer.py \
        --manifest /tmp/xasr-hotword-controls/manifest.tsv --model x-asr-zh-en-q8_0.gguf \
        --from-file "$run.tsv" --label "$run"
done
```

Run the same commands for the holdout and tuning sets. For HTP, cross-build
and deploy using [BUILD.md](BUILD.md), upload the WAVs and profile, and repeat
each CLI invocation with `HTP0:0`, polling enabled, and the device file paths.
A biased HTP invocation should report `joiner HTP0:0 7/7 ops`.

### Results on SM6650

The gain was selected on the Tingting tuning pack. On the host CPU backend,
gains 1, 2, and 3 gave Mandarin CER 7.83%, 7.23%, and 7.23%, compared with
8.43% without a profile. Gain 2 was the first configuration with improved
English WER in that sweep. The Eddy holdout changes only the Mandarin
voice, so its English clips are repeats rather than an independent English
evaluation.

| HTP pack | Count | Mandarin CER off → on | English WER off → on |
|---|---:|---:|---:|
| Tingting tuning | 32 | 8.43% → 7.23% | 5.34% → 4.58% |
| Eddy holdout | 32 | 44.58% → 44.58% | 5.34% → 4.58% |
| Homophone, ordinary speech, and polarity controls | 16 | 1.23% → 1.23% | 0.00% → 0.00% |

The tuning changes corrected `座以/座已` to `座椅` in two commands, and
`seat peter` to `seat heater` in one English command. The changed Mandarin
commands still said passenger seat heat level 2 and rear-left seat
ventilation level 1. The English level remained two and the baseline's
`said` error remained. The profile left all 16 control transcripts and
both silence transcripts unchanged. These include opposed actions with
left/right targets and a command that leaves air conditioning off while
turning on a specified light. The `氛围灯 → 分为灯` error remained.

HTP polling was on and the biased joiner reported 7/7 nodes on HTP. Median
unpaced RTF on the 32 tuning WAVs was 0.2417 without bias and 0.2432 with
bias in one pass per arm. The absolute timing difference is small and the
sample does not establish a latency regression or improvement. Use fewer
or weaker phrases if a dedicated field control set finds substitutions.
The profile stays opt-in. Real cabin audio and independent English speakers
are still needed to establish field accuracy.

The explicit CPU `--joiner-host` path reproduced the tuning CER/WER changes.
With the same profile in the HTP server, the following replay produced eight
transcripts identical to the biased HTP CLI. The server reported four
width-two dispatches and no width-four dispatch in this replay.

```sh
# Start llama-x-asr-server on Android with --streams 4 --encoder-batch 4
# and --hotwords pointing to the uploaded profile. Forward its port with adb.
python3 examples/x-asr/tools/asr_client.py --port 18775 --sessions 4 --reps 2 --json \
    --wavs "$TUNING/seat_zh01.wav,$TUNING/seat_en01.wav,$TUNING/seat_zh03.wav,$TUNING/win_zh03.wav"
```

### Field-level result, three rounds over four packs

The numbers above compare one pass per arm against CER. That is enough to rank
variants and not enough to claim a command works, so the same profile was
re-run through the field grader: three rounds per arm, arms alternating inside
each round, a warm-up clip first, and every clip of every pack in one
invocation.

```sh
python3 examples/x-asr/tools/bench_vehicle.py --rounds 3 --arms off,on \
    --sets tingting,holdout,controls,hardneg
```

It copies each pack's manifest off the phone, runs the arms itself, grades
every round, and writes `config.json`, `report.json` and `summary.md` with the
run identity: binary, model and profile hashes, device, and the exact
environment. Counts are rows, so each of the 32 tuning clips appears three
times; percentages are identical to per-clip rates.

| arm | pack | clips | device | state | value | clause | command | wrong dir | unstated | false trig | near miss | zh CER | en WER |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | Tingting tuning | 96 | 83.3% (105/126) | 76.2% (96/126) | 77.8% (42/54) | 76.2% (96/126) | 68.8% (66/96) | 0 | 9 | 0/0 | 0/0 | 8.43% | 5.34% |
| on | Tingting tuning | 96 | 90.5% (114/126) | 81.0% (102/126) | 94.4% (51/54) | 81.0% (102/126) | 75.0% (72/96) | 0 | 12 | 0/0 | 0/0 | 7.23% | 4.58% |
| off | Eddy holdout | 96 | 59.5% (75/126) | 45.2% (57/126) | 55.6% (30/54) | 45.2% (57/126) | 34.4% (33/96) | 3 | 15 | 0/0 | 0/0 | 44.58% | 5.34% |
| on | Eddy holdout | 96 | 61.9% (78/126) | 45.2% (57/126) | 61.1% (33/54) | 45.2% (57/126) | 34.4% (33/96) | 3 | 18 | 0/0 | 0/0 | 44.58% | 4.58% |
| off | controls | 48 | 100% (24/24) | 100% (24/24) | - | 100% (24/24) | 100% (12/12) | 0 | 0 | 0/36 | 3/36 | 1.23% | 0.00% |
| on | controls | 48 | 100% (24/24) | 100% (24/24) | - | 100% (24/24) | 100% (12/12) | 0 | 0 | 0/36 | 3/36 | 1.23% | 0.00% |
| off | hard negatives | 48 | - | - | - | - | - | 0 | 0 | 0/48 | 27/48 | 8.62% | 6.67% |
| on | hard negatives | 48 | - | - | - | - | - | 0 | 0 | 0/48 | 30/48 | 6.90% | 6.67% |

| arm | pack | wall ms p50 | p95 | p99 | RTF median |
|---|---|---:|---:|---:|---:|
| off | Tingting tuning | 544 | 697 | 708 | 0.216 |
| on | Tingting tuning | 546 | 701 | 712 | 0.216 |
| off | Eddy holdout | 543 | 696 | 708 | 0.216 |
| on | Eddy holdout | 546 | 701 | 712 | 0.217 |

`wall_ms` is the CLI's per-clip time for one whole VAD segment, unpaced, so it
is the endpoint-to-final interval without transport or queueing. It is not the
server's first-byte or completion latency.

**An 8.43% CER hides a 31.3% command error rate.** Without the profile, 10 of
32 tuning commands would not execute: three Mandarin seat commands
(`座以/座已` for 座椅), two interior-light commands (`分为灯` for 氛围灯), four
English commands whose `Set` became `said`, and one `seat heater` that became
`seat peter`.

The profile repairs nouns and leaves verbs alone. It fixes `座以/座已 → 座椅` on
two seat commands and `seat peter → seat heater`, which raises device hits from
105 to 114 of 126 clauses and value hits from 42 to 51 of 54. The four `Set →
said` commands still fail, including the one whose noun the profile just fixed,
because `said` carries no direction. Two clips (six rows) change verdict, from
22/32 to 24/32. The `unstated` count rises from 9 to 12 for the same reason:
more devices are recognised with no verb attached.

The control and hard-negative packs are where the profile could have done
damage, and it does not. All 12 control commands keep their direction, 0 of 36
control rows and 0 of 48 hard-negative rows become actionable, including
`Do not open the front right window, open the rear left window`,
`座椅加热别开，只打开座椅通风` and `Leave the air conditioning off`. The near-miss
count rises by one clip (`角窝灯 → 脚窝灯` on `脚窝灯有点暗`), so the boost is
active on statements too; the verb gate is what keeps them from executing.

It does not transfer to the second Mandarin voice. The Eddy pack fails 63 of 96
rows in both arms and its CER is identical at 44.58%: 把 becomes 马/买/脑袋,
座椅 becomes 手架/座后, 内循环 becomes `i 类循环`. One of those rows is a
direction flip that survives every round and both arms — `别开左后车窗` is
recognised as `解开左后车窗`, which opens the window the driver asked to keep
shut. A single synthetic voice is not evidence of field behaviour, and this is
the failure class that matters most.

Transcripts are identical across all three rounds and both arms for 91 of 96
clips. The five that move are the four seat clips the profile touches plus the
`角窝灯` statement. The same 32 tuning clips also transcribe byte-identically to
the pre-fold build in both arms, so the streaming cache change in this branch
is transcription-neutral on this set.

The profile costs 2 ms at p50 and 4 ms at p95 over 96 rows, which is inside the
run-to-run spread of a single arm; this sample does not establish a latency
cost. The profile stays opt-in.

The raw A/B TSVs, device placement logs, audio and profile hashes, and
source digest are in
`$HOME/Library/Caches/qwen3-asr-htp/xasr-multistream/cabin-hotwords.DRzGS6`.
The three-round field run, its per-round TSVs, `config.json` and `report.json`
are in
`$HOME/Library/Caches/qwen3-asr-htp/xasr-vehicle-20261117-115921`. The hard
negatives are in
`$HOME/Library/Caches/qwen3-asr-htp/xasr-multistream/vehicle-hardneg.3hSTPf`.

### Why the profile cannot repair `Set the`

The four English commands that still fail all begin `Set the <noun>`. Before
adding verbs to the profile, check what the joiner actually scored. The
diagnostic prints the five best candidates at every emitted token:

```sh
XASR_DEBUG_LOGITS=1 build-x-asr/bin/llama-x-asr x-asr-zh-en-q8_0.gguf CPU \
    "$TUNING/hvac_en01.wav" --joiner-host
```

| clip | winner | `▁Said` | `▁set` |
|---|---|---:|---:|
| `hvac_en01` `Set the driver temperature …` | `▁said` | −2.64 | **−6.38** |
| `hvac_en02` `Set the passenger temperature …` | `▁said` | −2.70 | **−5.70** |
| `seat_en01` `Set the passenger seat heater …` | `▁said` | −2.51 | **−6.42** |
| `seat_en03` `Set the rear left seat ventilation …` | `▁said` | −2.71 | **−1.76** |

Gaps are raw logits below the winner. `▁set` is not even the runner-up: the
vocabulary holds `▁set` (4405), `▁said` (4118) and `▁Said` (4881), and both
`said` forms outrank it, so this is not a capitalisation or tokenisation
mismatch — there is no `▁Set` token to aim at.

The bias cannot close that gap. A phrase's first piece receives 0.3 × gain and
the loader accepts at most 3, so the ceiling is **0.9 logits** against a needed
1.76–6.42. No entry, at any allowed gain, can flip these four tokens; adding
`▁set ▁the` to the profile would only raise the false-trigger surface. This is
why the field result above repairs nouns and not verbs.

The failure is a property of the audio-and-model pair, not of one fixture:

- **Position.** Same voice, one word changed. `Set the fan to level three` →
  `said` (`▁set` −1.88). `Set fan to level three` → `set` (−5.21). `Please set
  the fan …` → `set`. `Turn on recirculation and set the fan …` → `set`. Only
  the bare utterance-initial `Set the` fails.
- **Speaker.** Six synthesised voices, same sentence: Samantha, Daniel, Karen
  and Kathy produce `said`, Moira `Seth`, and Reed returns an empty transcript.
  The holdout pack, which uses a different Mandarin voice, fails the same five
  English clips.

```sh
for v in Samantha Daniel Karen Moira Kathy Reed; do
    say -v "$v" -o /tmp/set.aiff "Set the driver temperature to twenty one degrees"
    afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/set.aiff "/tmp/set-$v.wav"
done
for text in "Set the fan to level three" "Set fan to level three" \
            "Please set the fan to level three"; do
    say -v Samantha -o /tmp/pos.aiff "$text"
    afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/pos.aiff "/tmp/pos-${text// /_}.wav"
done
```

So the pattern is a real /t/-before-/ð/ reduction that the model resolves toward
the more common sentence-initial word. Repairing it needs something the current
mechanism cannot do: a decoder restricted to a closed verb set (a gain of ≥7
logits, i.e. a command grammar rather than a phrase list), an English acoustic
model, or post-hoc rescoring that rewrites the verb — the last one changes the
transcript and is a product decision, not a tuning knob.

### The bias is per stream

Bias vectors live in the per-stream decoder state and the phrase list is
read-only, so concurrent streams should not share a boost. The check runs each
clip alone and then all of them at one endpoint, and requires the same text:

```sh
python3 examples/x-asr/tools/bench_bias_isolation.py --port 18081 --rounds 3 \
    --wavs "$HN/neg_zh01.wav,$TUNING/seat_zh03.wav,$HN/neg_en01.wav,$TUNING/hvac_en01.wav"
```

Four streams (two commands, two statements that name devices) over three rounds:
**12 of 12 concurrent transcripts identical to their single-stream text**, with
`endpoint→final` 1249–1601 ms for four lanes committing at the same endpoint.
The statements stay non-actionable (`车窗外面风有点大`,
`The window is a little dirty.`) while the commands beside them keep their
device, direction and value, which is the case a leaked boost would break.

