#!/bin/sh
# Acceptance for a running deployment: contract, accuracy and latency in one go.
#
#   examples/paraformer/tools/accept.sh [PORT] [FIXTURE_DIR]
#
# Exits non-zero if any gate fails. This is the command that says "this box is
# deployable"; DEPLOY.md records what it printed on SM6650.
set -eu

PORT="${1:-8899}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
FIXTURES="${2:-${PF_MODELS:-$HOME/models/paraformer}/tingting}"
PYTHON="${PYTHON:-python3}"
RUN="${PF_RUN:-/tmp/pf-accept}"

[ -f "$FIXTURES/manifest.tsv" ] || { echo "no fixtures at $FIXTURES" >&2; exit 2; }
mkdir -p "$RUN"
WAV="$FIXTURES/hvac_zh01.wav"
[ -f "$WAV" ] || WAV=$(ls "$FIXTURES"/*_zh*.wav | head -1)

echo "== health"
"$PYTHON" - "$PORT" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
h = json.load(urllib.request.urlopen("http://127.0.0.1:%s/health" % port, timeout=5))
print("   ", json.dumps(h)[:200])
assert h.get("status") == "ok", "health is not ok: %s" % h.get("status")
assert h.get("htp_engines", 0) >= 1, "no HTP engine: %s" % h
PY

echo "== contract (smoke_production)"
"$PYTHON" "$HERE/tools/smoke_production.py" --port "$PORT" --wav "$WAV" | tail -3

echo "== accuracy through the OpenAI endpoint"
"$PYTHON" "$HERE/tools/eval_api.py" --port "$PORT" --wav-dir "$FIXTURES" --out "$RUN/api.tsv" --punctuate
"$PYTHON" "$HERE/../x-asr/tools/eval_vehicle.py" --pack "tingting=$FIXTURES/manifest.tsv" --run "api=$RUN/api.tsv" | tail -2

echo "== latency, one and two lanes"
"$PYTHON" "$HERE/tools/bench_gate_overhead.py" --port "$PORT" --wav "$WAV" --lanes 1 --runs 8 --label "1-lane"
"$PYTHON" "$HERE/tools/bench_gate_overhead.py" --port "$PORT" --wav "$WAV" --lanes 2 --runs 6 --label "2-lane"

echo "== accept ok (run dir: $RUN)"
