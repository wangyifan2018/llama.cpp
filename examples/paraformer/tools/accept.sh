#!/bin/sh
# Acceptance for a running deployment: contract, accuracy and latency in one go.
#
#   examples/paraformer/tools/accept.sh [PORT] [FIXTURE_DIR]
#
# Fail on a contract or synthetic command regression. Latency is recorded for
# comparison with the measured SM6650 baseline and the product SLO.
set -eu

PORT="${1:-8899}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
FIXTURES="${2:-${PF_MODELS:-$HOME/models/paraformer}/tingting}"
PYTHON="${PYTHON:-python3}"
RUN="${PF_RUN:-$(mktemp -d "${TMPDIR:-/tmp}/pf-accept.XXXXXX")}"

[ -f "$FIXTURES/manifest.tsv" ] || { echo "no fixtures at $FIXTURES" >&2; exit 2; }
mkdir -p "$RUN"
WAV="$FIXTURES/hvac_zh01.wav"
[ -f "$WAV" ] || { echo "missing reference WAV $WAV" >&2; exit 2; }
PACK=$(basename "$(cd "$FIXTURES" && pwd)")
FIXTURES=$(cd "$FIXTURES" && pwd)
BASE="http://127.0.0.1:$PORT"
WS="ws://127.0.0.1:$PORT/v1/realtime"

run_logged() {
    name=$1
    shift
    status=0
    "$@" >"$RUN/$name.log" 2>&1 || status=$?
    cat "$RUN/$name.log"
    return "$status"
}

(cd "$FIXTURES" && sha256sum -c "$HERE/tools/tingting-fixtures.sha256") >"$RUN/fixture-hashes.txt"

echo "== health"
"$PYTHON" - "$BASE" "$RUN" <<'PY'
import json, sys, urllib.request
from pathlib import Path
base, dest = sys.argv[1:]
with urllib.request.urlopen(base + "/health", timeout=5) as response:
    h = json.load(response)
Path(dest, "health.json").write_text(json.dumps(h, sort_keys=True, indent=2) + "\n")
if h["status"] != "ok" or h["htp_engines"] != 1:
    raise SystemExit(f"expected one ready HTP engine: {h}")
if h["engines"] != 2 or h["routing"] != "prefer_idle_htp":
    raise SystemExit(f"expected HTP plus CPU with HTP priority: {h}")
if not h["server_vad"] or h["speaker_gate"] or h["partials"]:
    raise SystemExit(f"expected VAD enabled, speaker and preview disabled: {h}")
PY

echo "== contract (smoke_production)"
run_logged smoke "$PYTHON" "$HERE/tools/smoke_production.py" --port "$PORT" --wav "$WAV" --expect-vad
if [ "${PF_REPLAY_PROTOCOL_CHECK:-0}" = 1 ]; then
    run_logged realtime-contract "$PYTHON" "$HERE/tools/check_realtime_vad.py" --port "$PORT" --wav "$WAV"
fi

echo "== accuracy through the OpenAI endpoint"
run_logged api "$PYTHON" "$HERE/tools/eval_api.py" --port "$PORT" --wav-dir "$FIXTURES" --out "$RUN/api.tsv" --punctuate
run_logged quality "$PYTHON" "$HERE/../x-asr/tools/eval_vehicle.py" --pack "$PACK=$FIXTURES/manifest.tsv" \
    --run "api=$RUN/api.tsv" --json "$RUN/quality.json"

echo "== REST latency"
for lanes in 1 2 4; do
    run_logged "rest-$lanes" "$PYTHON" "$HERE/tools/bench_gate_overhead.py" --port "$PORT" \
        --wav "$WAV" --lanes "$lanes" --runs 8 --label "rest-$lanes" --out "$RUN/rest-$lanes.jsonl"
done

echo "== Realtime latency (20 ms PCM packets, client commit)"
"$PYTHON" - "$WAV" "$RUN/mixed.wav" <<'PY'
import sys, wave
src, dest = sys.argv[1:]
with wave.open(src, "rb") as inp:
    assert inp.getparams()[:3] == (1, 2, 16000), src
    pcm = inp.readframes(inp.getnframes())
clip = pcm + bytes(9600)
with wave.open(dest, "wb") as out:
    out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
    out.writeframes((clip * 4)[:320000])
PY
realtime() {
    scenario=$1
    shift
    run_logged "ws-$scenario" "$PYTHON" "$HERE/tools/bench_ws_realtime.py" --url "$WS" \
        --metrics "$BASE/metrics.json" --rounds "${PF_WS_ROUNDS:-3}" --out "$RUN/ws-$scenario.jsonl" "$@"
}
realtime single --wav "$WAV" --sessions 1
realtime aligned --wav "$WAV" --sessions 4
realtime staggered --wav "$WAV" --sessions 4 --stagger-ms 300
realtime mixed --wavs "$RUN/mixed.wav" "$WAV" "$FIXTURES/light_en01.wav" "$FIXTURES/light_en03.wav" \
    --start-offsets-ms 0 7700 8050 7570

"$PYTHON" - "$RUN/quality.json" "$PACK" <<'PY'
import json, sys
from pathlib import Path
path, pack = sys.argv[1:]
score = json.loads(Path(path).read_text())["arms"]["api"][pack]
if score["command_clips"] != 32 or score["command_ok"] < 28 or score["wrong_direction"] != 0:
    raise SystemExit(f"Tingting command regression: {score}")
PY

echo "== accept ok (run dir: $RUN)"
