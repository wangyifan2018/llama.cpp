#!/system/bin/sh
# Run from the E36 device directory with the E28 and E35 models staged.
set -eu
baseline=/data/local/tmp/q3tts/mmproj-qwen3-tts-0.6b-f16.gguf
quant=/data/local/tmp/q3tts-spec-quant/all-q8.gguf
backbone=/data/local/tmp/q3tts/qwen3-tts-0.6b-q8.gguf
ref=/data/local/tmp/q3tts/ref.wav
tab=$(printf '\t')
export LD_LIBRARY_PATH=.:/data/local/tmp/q3tts-spec-20261122
export TTS_STREAM=1 TTS_TRACE=1 TTS_CHUNK=8 TTS_SPEC_PIN_MASK=f0
unset TTS_FAKE_NAR TTS_SPEC_CAPTURE TTS_VOC_RES
while IFS="$tab" read -r id lang seed order prompt; do
    [ "$id" = id ] && continue
    if [ "$order" = RQ ]; then first=original; second=q8; else first=q8; second=original; fi
    for mode in "$first" "$second"; do
        prefix="$id-$mode"
        if [ "$mode" = original ]; then mm=$baseline; else mm=$quant; fi
        echo "$prefix start $(date +%s)"
        timeout 240s taskset f0 ./llama-tts -m "$backbone" -mm "$mm" --no-mmproj-offload -ngl 0 \
          -t 4 --cpu-mask f0 --cpu-strict 1 --load-mode none --tts-lang "$lang" --tts-speaker-file "$ref" \
          --top-k 50 --top-p 0.8 --temp 1 --seed "$seed" -n 96 -p "$prompt" -o "$prefix.wav" >"$prefix.log" 2>&1
        test -s "$prefix.wav"
        echo "$prefix finish $(date +%s)"
    done
done < e2e-prompts.tsv
