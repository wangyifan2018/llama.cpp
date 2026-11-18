#!/bin/sh
# Start the ASR service on the phone and expose it on the host.
#
#   examples/paraformer/tools/serve_device.sh [port]
#
# Pushes the console page, then starts the service. Binaries, libs and models
# are expected to be in place already (that is the deployment step, not this
# script) -- but web/ is pushed every time, because a stale page silently keeps
# talking to an older event schema and looks like a server bug.
set -eu

PORT="${1:-8899}"
DIR=/data/local/tmp/llama.cpp
ADB="${ADB:-adb}"
HERE=$(cd "$(dirname "$0")" && pwd)

# The console ships with the binary, so push it rather than trusting whatever
# copy happens to be on the device.
if [ -d "$HERE/../web" ]; then
    $ADB shell -T "mkdir -p $DIR/web"
    $ADB push "$HERE/../web/." "$DIR/web/" </dev/null >/dev/null 2>&1 \
        || { echo "failed to push the console from $HERE/../web" >&2; exit 1; }
fi

$ADB shell -T "p=\$(pidof llama-paraformer-api); [ -n \"\$p\" ] && kill \$p; sleep 2"
# keep the SoC awake so the process is not reaped while testing
$ADB shell svc power stayon true >/dev/null 2>&1 || true
$ADB shell -T "cd $DIR && export LD_LIBRARY_PATH=\$PWD/lib:\$PWD/onnx/bin ADSP_LIBRARY_PATH=\$PWD/lib \
    GGML_HEXAGON_DEVICES=HTP0:0 GGML_HEXAGON_OPPOLL=1 && \
    nohup bin/llama-paraformer-api paraformer-zh-q8_0.gguf --device HTP0:0 --port $PORT \
        --engines 2 --threads 2 --web-dir web \
        --speaker-model onnx/model/spk_od.onnx --enroll-dir pf-enroll \
        --vad-model onnx/model/silero_vad.onnx \
        --warmup pf-api-warm.wav > pf-api.log 2>&1 & sleep 1"

$ADB forward "tcp:$PORT" "tcp:$PORT" >/dev/null

i=0
while [ $i -lt 40 ]; do
    if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    i=$((i + 1))
    sleep 1
done

echo "health: $(curl -s -m 5 "http://127.0.0.1:$PORT/health" || echo 'not up yet, check the device log')"
echo "console: http://127.0.0.1:$PORT/"
