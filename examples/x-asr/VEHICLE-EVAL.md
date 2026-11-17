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
prefix. Bias vectors are maintained per stream. The built-in profile gives
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

The raw A/B TSVs, device placement logs, audio and profile hashes, and
source digest are in
`$HOME/Library/Caches/qwen3-asr-htp/xasr-multistream/cabin-hotwords.DRzGS6`.
