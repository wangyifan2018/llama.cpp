#!/bin/sh
# End-to-end check: real PCM over the socket must give the same final text as
# the in-process CLI on the same file.
#
#   MODEL=x-asr-zh-en-q8_0.gguf tools/server_smoke.sh "zh.wav en.wav" [DEVICE] [PORT]
set -eu

MODEL=${MODEL:?set MODEL to the GGUF path}
WAVS=${1:?give the WAV files as one quoted argument}
DEVICE=${2:-CPU}
PORT=${3:-18777}
BIN=${BIN:-build-x-asr/bin}
SERVER=$BIN/llama-x-asr-server
CLI=$BIN/llama-x-asr
CLIENT=$(dirname "$0")/asr_client.py

"$SERVER" "$MODEL" "$DEVICE" --port "$PORT" --streams 2 >/tmp/xasr-server-smoke.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
sleep 6

status=0
for wav in $WAVS; do
    expected=$("$CLI" "$MODEL" "$DEVICE" "$wav" 2>/dev/null | awk -F'\t' '{print $NF}')
    got=$(python3 "$CLIENT" --port "$PORT" --wav "$wav" --sessions 1 --paced | sed -E 's/.*text=//')
    if [ "$expected" = "$got" ]; then
        echo "PASS $wav"
    else
        echo "FAIL $wav"
        echo "  cli:    $expected"
        echo "  server: $got"
        status=1
    fi
done
exit $status
