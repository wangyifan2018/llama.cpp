#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
    echo "usage: $0 LLAMA_TTS MODEL_DIR OUTPUT_DIR" >&2
    exit 2
fi
tts=$1
model=$2
out=$3
mkdir -p "$out"
out=$(cd "$out" && pwd)
for seed in 1234 5678; do
    for sample in en zh; do
        case "$sample" in
            en) lang=english; prompt='Hello, this is a test. The next train arrives in five minutes.' ;;
            zh) lang=chinese; prompt='你好，今天是个晴天。下一班列车将在五分钟后到站。' ;;
        esac
        capture="$out/$sample-$seed.bin"
        : > "$capture"
        env -u TTS_FAKE_NAR -u TTS_VOC_RES TTS_SPEC_CAPTURE="$capture" TTS_TRACE=1 TTS_CHUNK=8 \
            "$tts" -m "$model/qwen3-tts-0.6b-q8.gguf" -mm "$model/mmproj-qwen3-tts-0.6b-f16.gguf" \
            -ngl 0 --no-mmproj-offload --seed "$seed" -n 24 -t 4 --top-k 50 --top-p 0.8 --temp 1 \
            --tts-lang "$lang" --tts-speaker-file "$model/ref.wav" -p "$prompt" -o "$out/$sample-$seed.wav" \
            >"$out/$sample-$seed.log" 2>&1
        test -s "$capture"
    done
done
cat "$out"/en-1234.bin "$out"/zh-1234.bin "$out"/en-5678.bin "$out"/zh-5678.bin > "$out/frames.bin"
