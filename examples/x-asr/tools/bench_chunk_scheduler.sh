#!/usr/bin/env bash
# Run from the repository root with the Android X-ASR package already deployed.
set -euo pipefail

if [[ $# != 4 ]]; then
    echo "usage: $0 SERVER_BINARY SHORT_WAV LONG_WAV LONG2_WAV" >&2
    exit 2
fi

adb_bin=${ADB:-adb}
server_binary=$1
short_wav=$2
long_wav=$3
long2_wav=$4
remote=${XASR_REMOTE:-/data/local/tmp/llama.cpp}
port=${XASR_PORT:-18768}
out=${XASR_OUT:-$(mktemp -d /tmp/xasr-chunk-ab.XXXXXX)}
reps=${XASR_REPS:-3}
scenarios=${XASR_SCENARIOS:-single,aligned4,mixed4}
mkdir -p "$out"
echo "Results: $out"

"$adb_bin" push "$server_binary" "$remote/bin/llama-x-asr-server-chunk-exp"
"$adb_bin" shell -T "cd '$remote' &&
    chmod 755 bin/llama-x-asr-server-chunk-exp &&
    sha256sum bin/llama-x-asr-server-chunk-exp x-asr-zh-en-q8_0.gguf lib/libggml*.so" > "$out/device-sha256.txt"
shasum -a 256 "$short_wav" "$long_wav" "$long2_wav" > "$out/audio-sha256.txt"
git rev-parse HEAD > "$out/checkout.txt"
"$adb_bin" forward --no-rebind "tcp:$port" "tcp:$port"
remote_pid=
stop_server() {
    if [[ -n "$remote_pid" ]]; then
        "$adb_bin" shell "kill $remote_pid" >/dev/null 2>&1 || true
        remote_pid=
    fi
}
trap 'stop_server; "$adb_bin" forward --remove "tcp:$port" >/dev/null 2>&1 || true' EXIT

args=(--port "$port" --short "$short_wav" --long "$long_wav" --long2 "$long2_wav")
for arm in ${XASR_ARMS:-request chunk}; do
    flag=
    case $arm in
        request) ;;
        chunk) flag=--chunk-scheduler ;;
        batch2) flag="--encoder-batch 2" ;;
        batch4) flag="--encoder-batch 4" ;;
        *) echo "unknown arm: $arm" >&2; exit 2 ;;
    esac
    remote_pid=$("$adb_bin" shell -T "cd '$remote' && (
        LD_LIBRARY_PATH=\$PWD/lib ADSP_LIBRARY_PATH=\$PWD/lib GGML_HEXAGON_DEVICES=HTP0:0 \
        GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPBATCH=2048 XASR_TIMING=1 \
        bin/llama-x-asr-server-chunk-exp x-asr-zh-en-q8_0.gguf HTP0:0 \
        --bind 127.0.0.1 --port $port --streams 4 --queue 64 $flag \
        > chunk-ab-$arm.log 2>&1 < /dev/null & echo \$!
    )")
    [[ "$remote_pid" =~ ^[0-9]+$ ]]
    ready=false
    for ((i = 0; i < 30; ++i)); do
        if "$adb_bin" shell "grep -q 'listening on' '$remote/chunk-ab-$arm.log'"; then
            ready=true
            break
        fi
        sleep 1
    done
    if [[ "$ready" != true ]]; then
        "$adb_bin" shell "tail -20 '$remote/chunk-ab-$arm.log'"
        exit 1
    fi

    python3 examples/x-asr/tools/bench_service.py "${args[@]}" --reps 1 --scenarios single > "$out/$arm.warmup"
    python3 -u examples/x-asr/tools/bench_service.py "${args[@]}" \
        --reps "$reps" --scenarios "$scenarios" --out "$out/$arm.jsonl" | tee "$out/$arm.summary"
    "$adb_bin" pull "$remote/chunk-ab-$arm.log" "$out/$arm.log" >/dev/null
    stop_server
done

python3 - "$out" ${XASR_ARMS:-request chunk} <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
arms = sys.argv[2:]

def transcripts(arm):
    rows = [json.loads(line) for line in (root / f"{arm}.jsonl").read_text().splitlines()]
    return {(r["scenario"], r["rep"], r["session"]): r["text"] for r in rows}

reference = transcripts(arms[0])
for arm in arms[1:]:
    actual = transcripts(arm)
    if not reference or reference.keys() != actual.keys():
        raise SystemExit(f"different A/B workload: {arm}")
    different = [key for key in reference if reference[key] != actual[key]]
    if different:
        raise SystemExit(f"transcript mismatches in {arm}: {different}")
    print(f"{len(reference)} matching transcripts: {arms[0]} vs {arm}")
PY
