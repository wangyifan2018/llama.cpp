#!/bin/sh
# Host-only gate for a change, without a device.
#
#   examples/paraformer/tools/ci.sh            # check + build + eval
#   examples/paraformer/tools/ci.sh check      # tooling and pinned weights only
#   examples/paraformer/tools/ci.sh eval       # accuracy gates only
#
# Takes the same PF_MODELS / PF_RUN layout as reproduce.sh and leaves the run
# directory behind so a failure can be inspected. Exit code is the gate result.
set -eu

HERE=$(cd "$(dirname "$0")/.." && pwd)
STEP="${1:-all}"

export PF_MODELS="${PF_MODELS:-$HOME/models/paraformer}"
export PF_CACHE="${PF_CACHE:-$HOME/pf-cache}"
export PF_RUN="${PF_RUN:-$HOME/pf-runs/ci}"

run_check() {
    "$HERE/reproduce.sh" check
}

run_build() {
    cmake --build build-x-asr --target llama-paraformer llama-paraformer-api llama-paraformer-server -j 8
}

run_fixtures() {
    # Only when the run directory is empty: fixtures are deterministic, so a CI
    # cache can reuse them, but a fresh checkout must be able to make them.
    if [ ! -f "$PF_RUN/cabin-expanded/manifest.tsv" ]; then
        "$HERE/reproduce.sh" fixtures
    fi
}

run_eval() {
    run_fixtures
    "$HERE/reproduce.sh" eval
}

mkdir -p "$PF_RUN"
case "$STEP" in
    check) run_check ;;
    build) run_check; run_build ;;
    eval)  run_check; run_eval ;;
    all)   run_check; run_build; run_eval ;;
    *)     echo "usage: $0 [check|build|eval|all]" >&2; exit 2 ;;
esac
echo "ci: $STEP ok"
