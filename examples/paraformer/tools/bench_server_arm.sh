#!/usr/bin/env bash
# Measure one server configuration on the device: startup, memory, latency.
#
#   examples/paraformer/tools/bench_server_arm.sh LABEL ENGINES WAV
#
# Every arm is a fresh process, so init cost and the first-graph allocation are
# paid inside the arm rather than inherited from an earlier one. That is the
# point: a latency number taken on a warm process cannot be compared with one
# taken on a cold process, and mixing the two is how a "regression" gets
# invented.
set -euo pipefail

ADB=${ADB:-$(command -v adb || echo "$HOME/Library/Android/sdk/platform-tools/adb")}
DIR=${DIR:-/data/local/tmp/pkgtest}
PORT=${PORT:-18940}
LABEL=$1
ENGINES=$2
WAV=$3
RUNS=${RUNS:-5}

say() { printf '%s\n' "$*"; }

kill_all() {
    $ADB shell -T 'p=$(pidof llama-paraformer-api); [ -n "$p" ] && kill -9 $p; sleep 2' </dev/null >/dev/null 2>&1 || true
}

kill_all

# Cold start, timed to the first successful /health. This includes loading
# 237 MB of weights and, on the HTP arm, session open plus graph compile.
$ADB shell -T "cd $DIR && export LD_LIBRARY_PATH=\$PWD/vendor/lib64 ADSP_LIBRARY_PATH=\$PWD/vendor/lib64 \
    GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPPOLL=1 \
    PF_HOTWORDS=\$PWD/vendor/etc/paraformer/hotwords_vehicle.txt && \
    nohup ./vendor/bin/llama-paraformer-api vendor/etc/paraformer/paraformer-zh-q8_0.gguf \
        --port $PORT --engines $ENGINES --threads 2 > $LABEL.log 2>&1 &" </dev/null 2>/dev/null || true

$ADB forward "tcp:$PORT" "tcp:$PORT" >/dev/null 2>&1

start=$(date +%s)
ready=0
for _ in $(seq 1 90); do
    if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done
startup=$(( $(date +%s) - start ))
[ "$ready" = 1 ] || { say "$LABEL: did not come up; see $DIR/$LABEL.log"; exit 1; }

# Resident set after readiness. Taken before any request so it reflects the
# cost of the configuration, not of buffered audio.
mem=$($ADB shell -T "awk '/VmRSS/{print \$2}' /proc/\$(pidof llama-paraformer-api)/status" </dev/null | tr -d '\r')

# Latency over RUNS sequential requests on the same clip. Sequential on purpose:
# concurrency measures the queue, and this is measuring one inference.
lat=()
for i in $(seq 1 "$RUNS"); do
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
        -F "file=@$WAV" -F "response_format=json" -o "/tmp/arm_$LABEL.$i.json"
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    lat+=($((t1 - t0)))
done

# Report the median, not the mean: the first request in a cold process carries
# a one-off cost and a mean would smear it across every sample.
sorted=$(printf '%s\n' "${lat[@]}" | sort -n | tr '\n' ' ')
median=$(printf '%s\n' "${lat[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
text=$(python3 -c "import json;print(json.load(open('/tmp/arm_$LABEL.1.json')).get('text',''))" 2>/dev/null || echo '')

say "arm=$LABEL engines=$ENGINES startup_s=$startup rss_kb=$mem median_ms=$median samples=[${sorted% }]"
say "text=$text"

kill_all