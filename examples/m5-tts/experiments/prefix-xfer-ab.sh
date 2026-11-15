#!/usr/bin/env bash
# Experiment 3 (doc 21.6 #3): how much of the prefill round trip is the
# 32-row shared prefix that is still piped to the stage on every cache hit?
# Starts the server with M5_PREFIX_STATS=1, fires hit and miss requests,
# then collects the engine-side [m5] prefix-xfer split and the stage-side
# discard timing from the log.
set -u
ADB=${ADB:-adb}
V=/data/local/tmp/llama-m5-htp-v73
URL=http://127.0.0.1:18080
OUT=${M5_ASSETS:-$HOME/m5-assets}/prefix-xfer-results.txt
: > "$OUT"

$ADB shell 'for p in $(pidof m5-server); do kill $p; done; sleep 2;
    for n in backbone-resident depth-resident; do
        for p in $(pidof $n 2>/dev/null); do kill $p; done
    done; sleep 1' >/dev/null 2>&1

$ADB shell "cd $V && D=\$PWD; \
    export LD_LIBRARY_PATH=\$D ADSP_LIBRARY_PATH=\$D \
           GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1 M5_PREFIX_STATS=1; \
    nohup ./m5-server model --port 8080 --cwd \$D --threads 3 \
        --voice serena --voice-encoder \$D/codec_encoder.onnx \
        > \$D/m5-server.log 2>&1 &" >/dev/null 2>&1
for _i in $(seq 1 60); do
    if curl -s -m 2 --noproxy '*' "$URL/health" | grep -q '"ok"'; then sleep 2; break; fi
    sleep 1
done
curl -s -m 2 --noproxy '*' "$URL/health" | grep -q '"ok"' || { echo "server down" | tee -a "$OUT"; exit 1; }

req() {
    local label=$1 text=$2
    curl -sS -m 90 --noproxy '*' -D /tmp/prefix-hdr.txt -o /tmp/prefix-body.bin \
        -X POST "$URL/v1/audio/speech" -H 'Content-Type: application/json' \
        -d "{\"input\":\"$text\",\"voice\":\"serena\",\"response_format\":\"wav\"}" >/dev/null 2>&1
    local first elapsed bb
    first=$(grep -ai '^X-M5-First-Pcm-S' /tmp/prefix-hdr.txt | tr -d '\r' | awk '{print $2}')
    elapsed=$(grep -ai '^X-M5-Elapsed-S' /tmp/prefix-hdr.txt | tr -d '\r' | awk '{print $2}')
    bb=$(grep -ai '^X-M5-Backbone-S' /tmp/prefix-hdr.txt | tr -d '\r' | awk '{print $2}')
    echo "$label first=$first elapsed=$elapsed backbone=$bb" | tee -a "$OUT"
}

req "warm_hit_1" '前方三百米右转，随后靠左行驶。'
req "warm_hit_2" '今天天气不错，适合出门走走，路上小心。'
req "warm_hit_3" '请在下一个路口左转，然后直行两百米。'
req "warm_hit_4" '好的，我们马上出发，请系好安全带。'

echo "=== engine prefix-xfer lines ===" | tee -a "$OUT"
$ADB shell "grep -a 'prefix-xfer' $V/m5-server.log" | tee -a "$OUT"
echo "=== stage hit lines ===" | tee -a "$OUT"
$ADB shell "grep -a 'm5-stage prefix' $V/m5-server.log" | tee -a "$OUT"
echo "=== prefix row stats ===" | tee -a "$OUT"
$ADB shell "grep -a '\[m5\] prefix shared' $V/m5-server.log" | tail -8 | tee -a "$OUT"
echo "DONE" | tee -a "$OUT"
