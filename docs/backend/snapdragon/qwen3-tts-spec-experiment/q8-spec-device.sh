#!/system/bin/sh
set -eu
export LD_LIBRARY_PATH=/data/local/tmp/q3tts-spec-20261122 TTS_SPEC_PIN_MASK=f0 TTS_SPEC_FLASH=1
TTS_SPEC_WINDOW3=1 taskset f0 ./q8-rounds all-q8.gguf frame0.bin q8-window.scores \
    4 10 0 >q8-window.csv 2>q8-window.log
for block in 1 2; do
    if [ "$block" = 1 ]; then modes='whole 16 8 2'; else modes='2 8 16 whole'; fi
    for mode in $modes; do
        stem="q8-rounds-$mode-$block"
        TTS_SPEC_GREEDY_ROUNDS="$mode" timeout 120s taskset f0 ./q8-rounds all-q8.gguf phone-eval.bin "$stem.codes" \
            4 1 0 >"$stem.csv" 2>"$stem.log"
    done
done
