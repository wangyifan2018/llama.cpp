#!/usr/bin/env bash
# A/B the HTP flash-attn setting on one device, one port, one fixture.
#
# The two configurations cannot run at the same time (both want HTP0), so they
# are alternated on a single port: A B B A. Same-process comparisons are
# misleading here -- the first request in a process pays graph construction, so
# each arm gets its own server and its own warmup, and the two passes cancel
# thermal drift in whichever direction it happens to run.
#
# usage: ab_flash_attn.sh [runs] [lanes...]      (each pass costs ~2x warmup)
# With four restarts plus warmup this takes several minutes; run it in the
# background or split the lanes across invocations rather than fighting a
# shell timeout halfway through and losing both arms.
set -euo pipefail

ADB=~/Library/Android/sdk/platform-tools/adb
DEV=/data/local/tmp/pf
PORT=${PORT:-18884}
RUNS=${1:-6}
shift || true
LANES=${*:-1 2 4}
WAV=${WAV:-hvac_zh01.wav}
HERE=$(cd "$(dirname "$0")" && pwd)
LOCAL_WAV=${LOCAL_WAV:-$HOME/Library/Caches/qwen3-asr-htp/paraformer-models/fixtures-tingting/$WAV}

start() { # $1 = extra flags
    "$ADB" shell "cd $DEV && LD_LIBRARY_PATH=$DEV/lib ADSP_LIBRARY_PATH=$DEV/lib \
        ./llama-paraformer-api paraformer-zh-q8_0.gguf --device HTP0 --port $PORT \
        --bind 127.0.0.1 --engines 2 --threads 2 --warmup tin/$WAV --no-partials $1 \
        > ab.log 2>&1 &" </dev/null
    for _ in $(seq 1 90); do
        if curl -sf -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then return 0; fi
        sleep 1
    done
    echo "server did not come up with flags: $1" >&2
    "$ADB" shell "tail -20 $DEV/ab.log" </dev/null >&2
    return 1
}

stop() {
    # Kill by PID, not `pkill -f llama-paraformer-api`: the pattern also matches
    # the shell running the pkill, so the shell is killed first and the server
    # survives (the caller then sees exit 137 and no server). `pkill -x` does not
    # work either -- the kernel truncates comm to 15 chars, so "llama-paraformer"
    # never equals the full name.
    local p
    p=$("$ADB" shell 'ps -A -o PID,ARGS | grep "[l]lama-paraformer-api" | awk "{print \$1}"' </dev/null)
    if [ -n "$p" ]; then
        # shellcheck disable=SC2086
        "$ADB" shell "kill $p" </dev/null > /dev/null 2>&1 || true
    fi
    sleep 2
}

bench() { # $1 = label
    for L in $LANES; do
        python3 "$HERE/bench_gate_overhead.py" --port "$PORT" --wav "$LOCAL_WAV" \
            --lanes "$L" --runs "$RUNS" --label "$1-$L" 2>&1 | tail -1
    done
}

trap 'stop' EXIT
"$ADB" forward "tcp:$PORT" "tcp:$PORT" > /dev/null

for pass in 1 2; do
    if [ "$pass" = 1 ]; then order="fa nofa"; else order="nofa fa"; fi
    for arm in $order; do
        stop
        if [ "$arm" = fa ]; then start "--htp-flash-attn"; else start ""; fi
        echo "--- pass $pass arm $arm"
        curl -s -m 5 "http://127.0.0.1:$PORT/health" | python3 -c "
import json,sys
d=json.load(sys.stdin)
b=[x for x in d['backends'] if x['actual'].startswith('HTP')][0]
print('    fa=%s enc=%d/%d dec=%d/%d' % (b['flash_attn'], b['encoder_device_ops'], b['encoder_total_ops'], b['decoder_device_ops'], b['decoder_total_ops']))
"
        bench "$arm/p$pass"
    done
done