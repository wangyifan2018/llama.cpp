#!/bin/sh
# From a clean checkout to the console this repository measures, in one command.
#
#   examples/paraformer/tools/reproduce_deployment.sh [PORT]
#
# Needs: docker (for the Android cross build), adb with one device, and PF_MODELS
# holding the weights from tools/fetch_models.py. The device must already have the
# model, the sherpa/onnxruntime libraries and the web console under
# /data/local/tmp/llama.cpp; tools/install_local.sh sets that directory up.
#
# It rebuilds the Android binary, installs the launcher and the config, restarts
# the service, and then runs tools/accept.sh, which is the gate. Expected on
# SM6650 (DEPLOY.md section 4): 23/23 contract checks, command field 87.5%,
# one lane 280/287 ms, two lanes 292/306 ms.
set -eu

PORT="${1:-8899}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PF_MODELS="${PF_MODELS:-$HOME/models/paraformer}"
ADB="${ADB:-adb}"
RUN="${PF_RUN:-/tmp/pf-reproduce}"

say() { printf '\n== %s\n' "$*"; }
die() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

[ -f "$PF_MODELS/paraformer-zh-q8_0.gguf" ] || die "missing $PF_MODELS/paraformer-zh-q8_0.gguf (run tools/fetch_models.py)"
command -v docker >/dev/null 2>&1 || die "docker is required for the Android build"
"$ADB" get-state >/dev/null 2>&1 || die "no device: adb get-state failed"

say "host build"
cmake --build "$REPO/build-x-asr" --target llama-paraformer llama-paraformer-api -j 8

say "android build"
docker run --rm --volume "$REPO:/workspace" --volume "/tmp/sherpa:/sherpa:ro" \
    --workdir /workspace --platform linux/amd64 -u "$(id -u):$(id -g)" \
    ghcr.io/snapdragon-toolchain/arm64-android:v0.8 \
    bash -c 'cmake --build /workspace/build-android-xasr --target llama-paraformer-api -j 6'

say "install on the device"
"$ADB" push "$REPO/build-android-xasr/bin/llama-paraformer-api" /data/local/tmp/llama.cpp/bin/ >/dev/null
ADB="$ADB" PF_MODELS="$PF_MODELS" "$HERE/tools/install_local.sh"

say "acceptance"
PF_MODELS="$PF_MODELS" PF_RUN="$RUN" "$HERE/tools/accept.sh" "$PORT"

say "console: http://127.0.0.1:$PORT/"
