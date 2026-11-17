#!/bin/sh
# Same-arrival-sequence comparison of the ONNX CPU control against the GGUF HTP
# arms with different encoder batch caps. One arm at a time on the phone, the
# client on the host through ADB forward, identical WAVs and scenarios.
#
#   XASR_REPS=3 bash tools/bench_arrival_matrix.sh
#
# Environment: ANDROID_SERIAL, XASR_REMOTE, XASR_CACHE, XASR_ARMS, XASR_PORT.
set -eu

A=${ADB:-adb}
REMOTE=${XASR_REMOTE:-/data/local/tmp/llama.cpp}
PORT=${XASR_PORT:-18765}
REPS=${XASR_REPS:-3}
CACHE=${XASR_CACHE:-$HOME/Library/Caches/qwen3-asr-htp}
ARMS=${XASR_ARMS:-"onnx-cpu htp-b1 htp-b2 htp-b4"}
ONNX_DIR=$REMOTE/onnx/model
OUT=${XASR_OUT:-/tmp/xasr-arrival-$(date +%Y%m%d-%H%M%S)}
HERE=$(dirname "$0")
mkdir -p "$OUT"

SHORT="$CACHE/zh.wav"
SHORT_EN="$CACHE/en.wav"
LONG="$CACHE/cabin-asr-probe/long-perf/zh-10.wav"
LONG_EN="$CACHE/cabin-asr-probe/long-perf/en-20.wav"
for f in "$SHORT" "$SHORT_EN" "$LONG" "$LONG_EN"; do
    [ -f "$f" ] || { echo "missing fixture $f" >&2; exit 1; }
done

kserver() {
    $A shell -T 'for n in llama-x-asr-server onnx-server; do p=$(pidof $n); [ -n "$p" ] && kill $p; done; sleep 1' 2>/dev/null || true
}

start_arm() {
    kserver
    case "$1" in
        onnx-cpu)
            $A shell -T "cd $REMOTE && export LD_LIBRARY_PATH=$REMOTE/onnx/bin ONNX_TIMING=1 && \
                nohup onnx/bin/onnx-server $ONNX_DIR 2 --port $PORT --streams 4 > $REMOTE/arm.log 2>&1" &
            ;;
        htp-b1|htp-b2|htp-b4)
            width=$(echo "$1" | sed 's/htp-b//')
            $A shell -T "cd $REMOTE && export LD_LIBRARY_PATH=$REMOTE/lib ADSP_LIBRARY_PATH=$REMOTE/lib \
                GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_OPPOLL=1 XASR_TIMING=1 && \
                nohup bin/llama-x-asr-server x-asr-zh-en-q8_0.gguf HTP0:0 --port $PORT --streams 4 \
                    --encoder-batch $width > $REMOTE/arm.log 2>&1" &
            ;;
        *) echo "unknown arm $1" >&2; exit 1 ;;
    esac
    sleep 14
    $A shell -T "grep -E 'listening|engine ' $REMOTE/arm.log | tail -2"
}

stop_arm() {
    $A shell -T 'for n in llama-x-asr-server onnx-server; do p=$(pidof $n); [ -n "$p" ] && kill $p; done' 2>/dev/null || true
}

for arm in $ARMS; do
    echo "=== $arm ==="
    start_arm "$arm"
    python3 "$HERE/bench_arrival.py" --port "$PORT" --label "$arm" --reps "$REPS" \
        --short "$SHORT" --short-en "$SHORT_EN" --long "$LONG" --long-en "$LONG_EN" \
        --out "$OUT/$arm.jsonl" | tee "$OUT/$arm.summary"
    $A shell -T "cat $REMOTE/arm.log" > "$OUT/$arm.log" 2>/dev/null || true
    echo "widths: $(grep -o 'xasr-batch size=[0-9]' "$OUT/$arm.log" | sort | uniq -c | tr '\n' ' ')"
    stop_arm
done
echo "logs: $OUT"
