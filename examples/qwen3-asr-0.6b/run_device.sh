#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
    echo 'usage: run_device.sh BUILD MODEL PROJECTOR LIBCXX FIXTURES OUT ROUNDS' >&2
    exit 2
fi
build=$1
model=$2
projector=$3
libcxx=$4
fixtures=$5
out=$6
rounds=$7
adb=( "${ADB:-adb}" )
if [[ -n "${ANDROID_SERIAL:-}" ]]; then adb+=(-s "$ANDROID_SERIAL"); fi
if [[ ! "$rounds" =~ ^[1-9][0-9]?$ ]]; then
    echo 'ROUNDS must be in 1..99' >&2
    exit 2
fi
mkdir -p "$out"
bin="$build/bin"
files=("$bin/llama-qwen3-asr-batch" "$bin/libllama-common.so" "$bin/libllama.so" "$bin/libmtmd.so"
       "$bin/libggml.so" "$bin/libggml-base.so" "$bin/libggml-cpu.so" "$bin/libggml-hexagon.so"
       "$build/ggml/src/ggml-hexagon/libggml-htp-v73.so")
# The Snapdragon preset may link this backend even when the run selects HTP.
if [[ -f "$bin/libggml-opencl.so" ]]; then files+=("$bin/libggml-opencl.so"); fi
sha256sum "${files[@]}" "$model" "$projector" "$libcxx" "$fixtures"/* >"$out/host.sha256"
remote=$("${adb[@]}" shell -T 'mktemp -d /data/local/tmp/qwen-batch2.XXXXXX')
case "$remote" in /data/local/tmp/qwen-batch2.*) ;; *) exit 2 ;; esac
cleanup() {
    "${adb[@]}" shell -T "rm -rf '$remote'" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${adb[@]}" shell -T 'getprop ro.board.platform; getprop ro.build.fingerprint' >"$out/device.txt"
"${adb[@]}" push "${files[@]}" "$remote/"
"${adb[@]}" push "$model" "$remote/model.gguf"
"${adb[@]}" push "$projector" "$remote/projector.gguf"
"${adb[@]}" push "$libcxx" "$remote/libc++_shared.so"
"${adb[@]}" push "$fixtures/." "$remote/"
{
    for f in "${files[@]}"; do sha256sum "$f" | awk -v name="$(basename "$f")" '{print $1 "  " name}'; done
    sha256sum "$model" | awk '{print $1 "  model.gguf"}'
    sha256sum "$projector" | awk '{print $1 "  projector.gguf"}'
    sha256sum "$libcxx" | awk '{print $1 "  libc++_shared.so"}'
    (cd "$fixtures" && sha256sum *.wav *.tsv)
} >"$out/device-expected.sha256"
"${adb[@]}" push "$out/device-expected.sha256" "$remote/"
"${adb[@]}" shell -T "cd '$remote' && sha256sum -c device-expected.sha256 && chmod 755 llama-qwen3-asr-batch" >"$out/device-verified.txt"

run() {
    local slots=$1 trace=$2 prefix=$3 mode=$4
    "${adb[@]}" shell -T "cd '$remote' && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=\$PWD GGML_HEXAGON_DEVICES=HTP0:0 $mode ./llama-qwen3-asr-batch model.gguf projector.gguf HTP0:0 $slots $trace.tsv" \
        >"$out/$prefix.jsonl" 2>"$out/$prefix.log"
}

for ((round=1; round <= rounds; round++)); do
    for trace in simultaneous staggered refill; do
        order=(1 2)
        if ((round % 2 == 0)); then order=(2 1); fi
        for slots in "${order[@]}"; do
            echo "$trace slots=$slots round=$round"
            run "$slots" "$trace" "$trace.$slots.$round" ""
        done
    done
done
if [[ "${QWEN_PROFILE:-0}" == 1 ]]; then
    run 2 simultaneous profile "GGML_HEXAGON_PROFILE=1 GGML_SCHED_DEBUG=2"
fi
