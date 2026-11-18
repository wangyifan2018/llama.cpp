#!/bin/sh
# Install and start the service on a device without root.
#
#   examples/paraformer/tools/install_local.sh [--with-capabilities]
#
# Production runs the same launcher from /vendor under init (see
# deploy/paraformer-asr.rc and tools/package_device.sh). That needs root. This
# script puts the identical tree under /data/local/tmp/llama.cpp and keeps the
# launcher alive with a supervisor loop, which is the best persistence available
# without a writable /vendor.
#
#   --with-capabilities  also enable the server VAD and the speaker gate
#                        (about 120 MB more; this device is memory constrained)
set -eu

ADB="${ADB:-adb}"
DIR=/data/local/tmp/llama.cpp
HERE=$(cd "$(dirname "$0")/.." && pwd)
CAPS=0
[ "${1:-}" = "--with-capabilities" ] && CAPS=1

say() { printf '%s
' "$*"; }

say "push launcher, config and console"
$ADB shell -T "mkdir -p $DIR/bin $DIR/lib $DIR/state $DIR/web $DIR/onnx/model"
$ADB push "$HERE/deploy/paraformer-asr.sh" "$DIR/paraformer-asr.sh" >/dev/null
$ADB shell -T "chmod 0755 $DIR/paraformer-asr.sh"
$ADB push "$HERE/web/index.html" "$DIR/web/index.html" >/dev/null

conf=$HERE/deploy/paraformer-asr.local.conf
if [ "$CAPS" = 1 ]; then
    conf=$(mktemp)
    sed -e 's|^PF_VAD_MODEL=.*|PF_VAD_MODEL=/data/local/tmp/llama.cpp/onnx/model/silero_vad.onnx|' \
        -e 's|^PF_SPEAKER_MODEL=.*|PF_SPEAKER_MODEL=/data/local/tmp/llama.cpp/onnx/model/spk_od.onnx|' \
        -e 's|^PF_ENROLL_DIR=.*|PF_ENROLL_DIR=/data/local/tmp/llama.cpp/pf-enroll|' \
        "$HERE/deploy/paraformer-asr.local.conf" >"$conf"
fi
$ADB push "$conf" "$DIR/paraformer-asr.conf" >/dev/null
[ "$CAPS" = 1 ] && rm -f "$conf"

say "stop anything already running"
$ADB shell -T 'p=$(pidof llama-paraformer-api); [ -n "$p" ] && kill $p; sleep 2; p=$(pidof paraformer-asr.sh); [ -n "$p" ] && kill $p; sleep 1; true'

say "start the supervisor loop"
$ADB shell -T "cd $DIR && setsid nohup sh -c 'while :; do PF_CONF=$DIR/paraformer-asr.conf $DIR/paraformer-asr.sh; sleep 2; done' > $DIR/state/supervisor.log 2>&1 < /dev/null & sleep 1; echo supervisor started"

i=0
while [ $i -lt 45 ]; do
    if $ADB shell -T 'curl -sf -m 2 http://127.0.0.1:8899/health >/dev/null 2>&1 && echo up' | grep -q up; then
        break
    fi
    i=$((i + 1))
    sleep 1
done

$ADB forward tcp:8899 tcp:8899 >/dev/null 2>&1 || true
say "health: $(curl -s -m 5 http://127.0.0.1:8899/health 2>/dev/null || echo 'not up; see /data/local/tmp/llama.cpp/state/paraformer-asr.log')"
say "console: http://127.0.0.1:8899/"
