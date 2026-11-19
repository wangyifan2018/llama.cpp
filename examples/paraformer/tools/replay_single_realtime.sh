#!/bin/sh
# Compare manual commit with configured server VAD thresholds on one Mandarin stream.
set -eu

PORT=${1:?usage: replay_single_realtime.sh PORT TINGTING_FIXTURES [manual|SILENCE_MS ...]}
FIXTURES=${2:?missing Tingting fixtures}
shift 2
if [ "$#" -eq 0 ]; then set -- manual 300; fi
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
OUT=${PF_RUN:-$(mktemp -d "${TMPDIR:-/tmp}/pf-single.XXXXXX")}
FIXTURES=$(cd "$FIXTURES" && pwd)
PACK=$(basename "$FIXTURES")
mkdir -p "$OUT"
(cd "$FIXTURES" && sha256sum -c "$HERE/tingting-fixtures.sha256") >"$OUT/fixture-hashes.txt"
awk -F '\t' '$2 == "zh"' "$FIXTURES/manifest.tsv" >"$OUT/mandarin.tsv"

run_arm() {
    arm=$1
    shift
    case "$arm" in
        manual) ;;
        ''|*[!0-9]*) echo "expected manual or a silence duration in ms" >&2; return 2 ;;
        *) set -- --server-vad --silence-ms "$arm" ;;
    esac
    run="$OUT/$arm"
    mkdir -p "$run"
    "$PYTHON" "$HERE/eval_realtime.py" --url "ws://127.0.0.1:$PORT/v1/realtime" \
        --manifest "$FIXTURES/manifest.tsv" --language zh --out "$run/realtime.tsv" --details "$run/realtime.jsonl" "$@"
    "$PYTHON" "$HERE/../../x-asr/tools/eval_vehicle.py" --pack "$PACK=$OUT/mandarin.tsv" \
        --run "ws=$run/realtime.tsv" --json "$run/quality.json"
}

for arm do
    run_arm "$arm"
done
