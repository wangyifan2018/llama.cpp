#!/bin/bash
# Doc 24: the server's wav path hard-codes set_chunking(4,16). That split was
# tuned when codec was the bottleneck; after the depth KV32 change the two
# branches are within 1.15% and 16-frame codec bursts leave the pipeline idle.
# Sweep (first, chunk) on the real product path, alternating rounds.
set -u
ADB=${ADB:-adb}
V=/data/local/tmp/llama-m5-htp-v73
OUT=${M5_ASSETS:-$HOME/m5-assets}/wavchunk-results.txt
HDR=/tmp/wc-hdr.txt
TEXT='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
: > "$OUT"

stop_server () {
  $ADB shell 'for p in $(pidof m5-server); do kill $p; done; sleep 2; for n in backbone-resident depth-resident; do for p in $(pidof $n 2>/dev/null); do kill $p; done; done; sleep 1' </dev/null
}

start_server () {
  local first=$1 chunk=$2
  $ADB shell "cd $V && D=\$PWD; export LD_LIBRARY_PATH=\$D ADSP_LIBRARY_PATH=\$D \
      GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1 \
      M5_WAV_FIRST=$first M5_WAV_CHUNK=$chunk; \
      nohup ./m5-server model --port 8080 --cwd \$D --threads 3 --voice serena \
      --voice-encoder \$D/codec_encoder.onnx > \$D/wc-server.log 2>&1 &" </dev/null
  # warmup + first request can take a few seconds
  for _ in $(seq 1 40); do
    sleep 1
    if curl -s -m 2 --noproxy '*' http://127.0.0.1:18080/health >/dev/null 2>&1; then return 0; fi
  done
  echo "server did not come up for $first/$chunk" >&2
  return 1
}

hit () {
  local tag=$1
  curl -s -m 120 --noproxy '*' -D "$HDR" -o /tmp/wc.wav \
    -H 'Content-Type: application/json' \
    -d "{\"input\":\"$TEXT\",\"voice\":\"serena\",\"response_format\":\"wav\"}" \
    http://127.0.0.1:18080/v1/audio/speech
  local el fr fp cd dp bb h
  el=$(grep -i '^X-M5-Elapsed-S:'    "$HDR" | tr -d '\r' | awk '{print $2}')
  fr=$(grep -i '^X-M5-Frames:'       "$HDR" | tr -d '\r' | awk '{print $2}')
  fp=$(grep -i '^X-M5-First-Pcm-S:'  "$HDR" | tr -d '\r' | awk '{print $2}')
  cd=$(grep -i '^X-M5-Codec-S:'      "$HDR" | tr -d '\r' | awk '{print $2}')
  dp=$(grep -i '^X-M5-Depth-S:'      "$HDR" | tr -d '\r' | awk '{print $2}')
  bb=$(grep -i '^X-M5-Backbone-S:'   "$HDR" | tr -d '\r' | awk '{print $2}')
  h=$(shasum -a 256 /tmp/wc.wav | cut -c1-12)
  local pf
  pf=$(awk -v e="${el:-0}" -v f="${fr:-0}" 'BEGIN{ if (f>0) printf "%.2f", e*1000/f; else print "NA" }')
  echo "$tag elapsed=${el:-NA} frames=${fr:-NA} first=${fp:-NA} bb=${bb:-NA} dp=${dp:-NA} cd=${cd:-NA} ms_per_frame=$pf wav=$h" | tee -a "$OUT"
}

run_combo () {
  local first=$1 chunk=$2 round=$3 n=${4:-3}
  stop_server || true
  start_server "$first" "$chunk" || return 1
  for i in $(seq 1 "$n"); do hit "r$round:$first/$chunk:$i"; done
}

ORDER_A="4 16"
ORDER_B="2 4"
ORDER_C="2 16"
ORDER_D="4 8"

for r in 1 2; do
  if [ $((r % 2)) -eq 1 ]; then
    run_combo 4 16 $r; run_combo 2 4 $r; run_combo 2 16 $r; run_combo 4 8 $r
  else
    run_combo 4 8 $r; run_combo 2 16 $r; run_combo 2 4 $r; run_combo 4 16 $r
  fi
done
echo DONE | tee -a "$OUT"
