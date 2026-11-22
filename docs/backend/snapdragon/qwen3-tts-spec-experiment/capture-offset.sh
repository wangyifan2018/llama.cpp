#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
    echo "usage: $0 LLAMA_TTS MODEL_DIR OUTPUT_DIR" >&2
    exit 2
fi
tts=$1
models=$2
out=$3
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
while IFS=$'\t' read -r split id lang seed prompt; do
    [ "$split" = split ] && continue
    file="$out/$id"
    : > "$file.bin"
    env -u TTS_FAKE_NAR -u TTS_VOC_RES TTS_SPEC_CAPTURE="$file.bin" TTS_CHUNK=32 \
        "$tts" -m "$models/qwen3-tts-0.6b-q8.gguf" -mm "$models/mmproj-qwen3-tts-0.6b-f16.gguf" \
        -ngl 0 --no-mmproj-offload --seed "$seed" -n 40 -t 4 --top-k 50 --top-p 0.8 --temp 1 \
        --tts-lang "$lang" --tts-speaker-file "$models/ref.wav" -p "$prompt" -o "$file.wav" >"$file.log" 2>&1
    test -s "$file.bin"
done < "$here/offset-prompts.tsv"
