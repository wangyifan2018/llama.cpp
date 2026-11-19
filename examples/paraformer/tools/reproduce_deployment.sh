#!/bin/sh
# Build the SM6650 package and replay it in an isolated Android directory.
set -eu

HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PORT=${1:-18804}
case "$PORT" in ''|*[!0-9]*) echo "expected a device TCP port" >&2; exit 2 ;; esac
PF_MODELS=${PF_MODELS:-$HOME/models/paraformer}
PF_SHERPA_SDK=${PF_SHERPA_SDK:-/tmp/sherpa}
PF_ONNX_DIR=${PF_ONNX_DIR:-$PF_SHERPA_SDK/lib}
export PF_MODELS PF_ONNX_DIR
ADB=${ADB:-adb}
if ! command -v "$ADB" >/dev/null 2>&1 && [ -x "$HOME/Library/Android/sdk/platform-tools/adb" ]; then
    ADB=$HOME/Library/Android/sdk/platform-tools/adb
fi
RUN=${PF_RUN:-$(mktemp -d "${TMPDIR:-/tmp}/pf-replay.XXXXXX")}
mkdir -p "$RUN"
RUN=$(cd "$RUN" && pwd)
printf 'run directory: %s\n' "$RUN"
FIXTURES=${PF_FIXTURES:-$PF_MODELS/tingting}
BUILD=$REPO/build-android-xasr
export BUILD

[ -r "$PF_SHERPA_SDK/inc/c-api.h" ] || { echo "missing Sherpa SDK under $PF_SHERPA_SDK" >&2; exit 2; }
if [ ! -r "$REPO/CMakeUserPresets.json" ]; then
    cp "$REPO/docs/backend/snapdragon/CMakeUserPresets.json" "$REPO/CMakeUserPresets.json"
fi
docker run --rm -v "$REPO:/workspace" -v "$PF_SHERPA_SDK:/sherpa:ro" \
    -w /workspace --platform linux/amd64 -u "$(id -u):$(id -g)" \
    ghcr.io/snapdragon-toolchain/arm64-android:v0.8 sh -ec '
        if [ ! -f build-android-xasr/CMakeCache.txt ]; then
            cmake --preset arm64-android-snapdragon-release -B build-android-xasr \
                -DLLAMA_BUILD_EXAMPLES=ON -DXASR_SPEAKER=ON \
                -DXASR_SHERPA_INCLUDE_DIR=/sherpa/inc -DXASR_SHERPA_LIB_DIR=/sherpa/lib
        fi
        cmake --build build-android-xasr --target llama-paraformer-api -j 6
    ' >"$RUN/build.log" 2>&1 || { cat "$RUN/build.log" >&2; exit 1; }
docker image inspect --format '{{index .RepoDigests 0}}' ghcr.io/snapdragon-toolchain/arm64-android:v0.8 >"$RUN/toolchain.txt"
(cd "$REPO" && git ls-files -c -o --exclude-standard -- examples/paraformer docs/backend/snapdragon/api-reference.md |
    tar -czf "$RUN/paraformer-source.tar.gz" -T -)
"$HERE/tools/package_device.sh" "$RUN/package" >"$RUN/package.log" 2>&1 || { cat "$RUN/package.log" >&2; exit 1; }
[ "${PF_PACKAGE_ONLY:-0}" != 1 ] || exit 0

processes=$("$ADB" shell -T 'ps -A -o ARGS')
htp_owner=$(printf '%s\n' "$processes" | awk 'index($0, "llama-paraformer-api") && index($0, "--device HTP") { print "yes"; exit }')
if [ "$htp_owner" = yes ]; then
    echo "an existing Paraformer process owns HTP. Run tools/accept.sh on that service or use a reserved device." >&2
    exit 1
fi
"${PYTHON:-python3}" -c 'import websockets' || { echo "install websockets for the selected PYTHON" >&2; exit 2; }

remote=$("$ADB" shell -T 'mktemp -d /data/local/tmp/pf-replay.XXXXXX')
case "$remote" in /data/local/tmp/pf-replay.*) ;; *) exit 2 ;; esac
forward= pid=
cleanup() {
    if [ -n "$pid" ]; then
        "$ADB" shell -T "kill -TERM $pid" >/dev/null 2>&1 || true
        for _ in $(seq 1 10); do
            if ! "$ADB" shell -T "kill -0 $pid" >/dev/null 2>&1; then break; fi
            sleep .2
        done
        "$ADB" shell -T "kill -KILL $pid" >/dev/null 2>&1 || true
        "$ADB" pull "$remote/state/api.log" "$RUN/device.log" >/dev/null 2>&1 || true
    fi
    if [ -n "$forward" ]; then "$ADB" forward --remove "tcp:$forward" >/dev/null 2>&1 || true; fi
    "$ADB" shell -T "rm -rf '$remote'" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$ADB" push "$RUN/package/." "$remote/"
"$ADB" shell -T "cd '$remote' && sha256sum -c SHA256SUMS" >"$RUN/device-hashes.txt"
config=$RUN/replay.conf
sed "s|/vendor/|$remote/system/vendor/|g" "$HERE/deploy/paraformer-asr.conf" >"$config"
cat >>"$config" <<EOF

PF_WORKDIR=$remote/state
PF_LOG=$remote/state/api.log
PF_PORT=$PORT
PF_WEB_DIR=$remote/system/vendor/etc/paraformer/web
PF_EXTRA_ARGS="--htp-warmup-ms 15000 --htp-watchdog-ms 5000"
EOF
"$ADB" push "$config" "$remote/replay.conf"
forward=$("$ADB" forward tcp:0 "tcp:$PORT")
if curl -s --max-time .5 "http://127.0.0.1:$forward/health" >/dev/null; then
    echo "device port $PORT already serves HTTP" >&2
    exit 1
fi
pid=$("$ADB" shell -T "cd '$remote' && (PF_CONF='$remote/replay.conf' sh system/vendor/bin/paraformer-asr >launcher.log 2>&1 </dev/null & echo \$!)")
case "$pid" in ''|*[!0-9]*) exit 2 ;; esac
for _ in $(seq 1 40); do
    if curl -fs --max-time 0.5 "http://127.0.0.1:$forward/health" >"$RUN/initial-health.json"; then break; fi
    if ! "$ADB" shell -T "kill -0 $pid" >/dev/null 2>&1; then break; fi
    sleep 0.5
done
[ -s "$RUN/initial-health.json" ] || { echo "replay did not become ready, see $RUN/device.log" >&2; exit 1; }
PF_RUN="$RUN/results" PF_REPLAY_PROTOCOL_CHECK=1 "$HERE/tools/accept.sh" "$forward" "$FIXTURES"
