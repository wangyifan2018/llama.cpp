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

# Batch 2 adds a scheduling dimension. QWEN_SCHEDULE picks the policy and
# QWEN_BUDGET_MS bounds one round of prefill (only used by "interleaved").
EXTRA_ENV="${QWEN_EXTRA_ENV:-}"
SCHEDULE="${QWEN_SCHEDULE:-blocking}"
BUDGET_MS="${QWEN_BUDGET_MS:--1}"

run() {
    local variant=$1 trace=$2 prefix=$3 profile_env=$4
    local slots=2 policy=eager conv=im2col
    case "$variant" in
        1) slots=1 ;;
        2) ;;
        windowed) policy=windowed ;;
        weight-first) conv=weight-first ;;
        windowed-weight-first) policy=windowed; conv=weight-first ;;
        packed) policy=packed ;;
        gather) conv=gather ;;
        packed-gather) policy=packed; conv=gather ;;
    esac
    "${adb[@]}" shell -T "cd '$remote' && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=\$PWD GGML_HEXAGON_DEVICES=HTP0:0 MTMD_QWEN3A_CONV=$conv $EXTRA_ENV $profile_env ./llama-qwen3-asr-batch model.gguf projector.gguf HTP0:0 $slots $trace.tsv 96 $policy $SCHEDULE $BUDGET_MS" \
        >"$out/$prefix.jsonl" 2>"$out/$prefix.log"
}

modes=(1 2)
case "${QWEN_PHASE2:-0}" in
    0) ;;
    1) modes+=(windowed weight-first windowed-weight-first) ;;
    packed) modes+=(packed) ;;
    gather) modes+=(packed gather packed-gather) ;;
    *) echo 'QWEN_PHASE2 must be 0, 1, packed, or gather' >&2; exit 2 ;;
esac
for ((round=1; round <= rounds; round++)); do
    for trace in simultaneous staggered refill; do
        for ((i=0; i<${#modes[@]}; i++)); do
            index=$i
            if ((round % 2 == 0)); then index=$((${#modes[@]} - 1 - i)); fi
            variant=${modes[index]}
            echo "$trace $variant round=$round"
            run "$variant" "$trace" "$trace.$variant.$round" ""
        done
    done
done
if [[ "${QWEN_PROFILE:-0}" == 1 ]]; then
    run 2 simultaneous profile "GGML_HEXAGON_PROFILE=1 GGML_SCHED_DEBUG=2"
    case "${QWEN_PHASE2:-0}" in
        1) run weight-first simultaneous profile.weight-first "GGML_HEXAGON_PROFILE=1 GGML_SCHED_DEBUG=2" ;;
        packed) run packed simultaneous profile.packed "GGML_HEXAGON_PROFILE=1 GGML_SCHED_DEBUG=2" ;;
        gather) run packed-gather simultaneous profile.packed-gather "GGML_HEXAGON_PROFILE=1 GGML_SCHED_DEBUG=2" ;;
    esac
fi
