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
