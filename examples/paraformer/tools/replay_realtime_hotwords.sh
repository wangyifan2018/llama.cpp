#!/bin/sh
# Replay one configuration over the Realtime API. Give each arm its own PF_RUN.
set -eu

PORT=${1:?usage: replay_realtime_hotwords.sh PORT HOTWORD_FIXTURES TINGTING_FIXTURES}
PACKS=${2:?missing hotword fixtures}
TINGTING=${3:?missing Tingting fixtures}
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
OUT=${PF_RUN:-$(mktemp -d "${TMPDIR:-/tmp}/pf-hotwords.XXXXXX")}
mkdir -p "$OUT"

replay() {
    name=$1
    fixtures=$2
    sums=$3
    (cd "$fixtures" && sha256sum -c "$HERE/$sums") >"$OUT/$name-fixture-hashes.txt"
    "$PYTHON" "$HERE/eval_realtime.py" --url "ws://127.0.0.1:$PORT/v1/realtime" \
        --server-vad --lanes 4 --stagger-ms 220 --manifest "$fixtures/manifest.tsv" \
        --out "$OUT/$name.tsv" --details "$OUT/$name.jsonl"
}

for pack in hotword-tts hotword-voices hotword-near; do
    annotations=hotword_annotations.tsv
    sums=hotword-fixtures.sha256
    case "$pack" in
        hotword-voices) sums=hotword-voices-fixtures.sha256 ;;
        hotword-near) sums=hotword-near-fixtures.sha256; annotations=hotword_near_annotations.tsv ;;
    esac
    replay "$pack" "$PACKS/$pack" "$sums"
    "$PYTHON" "$HERE/score_realtime_hotwords.py" --annotations "$HERE/$annotations" \
        --run "$OUT/$pack.jsonl" --out "$OUT/$pack-score.json"
done

replay tingting "$TINGTING" tingting-fixtures.sha256
"$PYTHON" "$HERE/../../x-asr/tools/eval_vehicle.py" --pack "tingting=$TINGTING/manifest.tsv" \
    --run "ws=$OUT/tingting.tsv" --json "$OUT/tingting-score.json"
