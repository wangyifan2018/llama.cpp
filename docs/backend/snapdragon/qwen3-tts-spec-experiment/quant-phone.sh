#!/usr/bin/env bash
# Requires the E28 Android CPU libraries and F16 mmproj on the device.
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo "usage: $0 ADB REPLAY_BINARY HOST_RUN_DIR DEVICE_RUN_DIR" >&2
    exit 2
fi
adb=$1
binary=$2
run=$3
device=$4
"$adb" shell "mkdir -p $device"
"$adb" push "$binary" "$device/replay"
if [ "${QUANT_KEEP_DOWN_ONLY:-0}" = 1 ]; then
    "$adb" push "$run/keep-down-q8.gguf" "$run/frame0.bin" "$run/phone-eval.bin" "$device/"
else
    "$adb" push "$run"/{down-f16,down-q8,all-q8}.gguf "$run/frame0.bin" "$run/phone-eval.bin" "$device/"
fi
"$adb" shell "cd $device && chmod 755 replay"
for block in 1 2; do
    if [ "$block" = 1 ]; then
        modes=(original down-f16 down-q8 all-q8)
        frames=phone-eval.bin
    else
        modes=(all-q8 down-q8 down-f16 original)
        frames=frame0.bin
    fi
    if [ "${QUANT_KEEP_DOWN_ONLY:-0}" = 1 ]; then modes=(keep-down-q8); fi
    for mode in "${modes[@]}"; do
        mm="$mode.gguf"
        if [ "$mode" = original ]; then mm=/data/local/tmp/q3tts/mmproj-qwen3-tts-0.6b-f16.gguf; fi
        stem="phone-$mode-$block"
        "$adb" shell "cd $device && LD_LIBRARY_PATH=/data/local/tmp/q3tts-spec-20261122 \
            TTS_SPEC_FLASH=1 TTS_SPEC_PIN_MASK=f0 TTS_SPEC_GROUPED=ar-greedy \
            taskset f0 ./replay $mm $frames $stem.scores 4 24 0 >$stem.csv 2>$stem.log"
        "$adb" pull "$device/$stem.csv" "$device/$stem.log" "$device/$stem.scores" "$run/"
    done
done
