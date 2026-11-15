#!/usr/bin/env bash
# Multi-entry codec prime cache: alternating A/B on the device.
#   cap=0  cache disabled (every request pays the prime)  -> cold baseline / hash oracle
#   cap=1  old single-entry behaviour
#   cap=4  new default
# The sequence serena -> official -> serena -> official -> serena exposes the
# single-entry eviction: with cap=1 the third and fifth requests are cold again.
set -u
ADB=${ADB:-adb}
V=/data/local/tmp/llama-m5-htp-v73
URL=http://127.0.0.1:18080
OUT=${M5_ASSETS:-$HOME/m5-assets}/prime-ab-results.txt
LOGDIR=${M5_ASSETS:-$HOME/m5-assets}/prime-ab-logs
mkdir -p "$LOGDIR"
: > "$OUT"

TEXT='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'

start_server() {
    local cap=$1
    $ADB shell 'for p in $(pidof m5-server); do kill $p; done; sleep 2;
        for n in backbone-resident depth-resident; do
            for p in $(pidof $n 2>/dev/null); do kill $p; done
        done; sleep 1' >/dev/null 2>&1
    $ADB shell "cd $V && D=\$PWD; \
        export LD_LIBRARY_PATH=\$D ADSP_LIBRARY_PATH=\$D \
               GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1 \
               M5_PRIME_CACHE=$cap M5_PRIME_STATS=1; \
        nohup ./m5-server model --port 8080 --cwd \$D --threads 3 \
            --voice serena --voice-encoder \$D/codec_encoder.onnx \
            > \$D/m5-server.log 2>&1 &" >/dev/null 2>&1
    for _i in $(seq 1 60); do
        if curl -s -m 2 --noproxy '*' "$URL/health" | grep -q '"ok"'; then sleep 1; return 0; fi
        sleep 1
    done
    echo "server did not come up (cap=$cap)" >&2
    return 1
}

req() {
    local label=$1 voice=$2 fmt=$3 cap=$4
    curl -sS -m 90 --noproxy '*' -D /tmp/prime-hdr.txt -o /tmp/prime-body.bin \
        -X POST "$URL/v1/audio/speech" -H 'Content-Type: application/json' \
        -d "{\"input\":\"$TEXT\",\"voice\":\"$voice\",\"response_format\":\"$fmt\"}" \
        >/dev/null 2>&1
    local first=-1 elapsed=-1 rtf=-1 frames=-1 sha=none
    if [ "$fmt" = wav ]; then
        first=$(grep -ai '^X-M5-First-Pcm-S' /tmp/prime-hdr.txt | tr -d '\r' | awk '{print $2}')
        elapsed=$(grep -ai '^X-M5-Elapsed-S' /tmp/prime-hdr.txt | tr -d '\r' | awk '{print $2}')
        rtf=$(grep -ai '^X-M5-RTF' /tmp/prime-hdr.txt | tr -d '\r' | awk '{print $2}')
        frames=$(grep -ai '^X-M5-Frames' /tmp/prime-hdr.txt | tr -d '\r' | awk '{print $2}')
        sha=$(shasum -a 256 /tmp/prime-body.bin | cut -c1-16)
    else
        local m
        m=$(curl -s -m 6 --noproxy '*' "$URL/metrics" | python3 -c \
            'import sys,json; d=json.load(sys.stdin); x=d.get("last") or {}; print(x.get("first_pcm_s"),x.get("elapsed_s"),x.get("rtf"),x.get("frames"))' 2>/dev/null)
        read -r first elapsed rtf frames <<< "$m"
    fi
    local line="$label cap=$cap voice=$voice fmt=$fmt first=$first elapsed=$elapsed rtf=$rtf frames=$frames sha=$sha"
    echo "$line" | tee -a "$OUT"
}

round=0
# Alternating order controls thermal drift: 0,1,4 then reverse.
for cap in 0 1 4 4 1 0; do
    round=$((round + 1))
    echo "=== round $round cap=$cap ===" | tee -a "$OUT"
    start_server "$cap" || { echo "ABORT" | tee -a "$OUT"; exit 1; }
    req "r${round}_s1" serena   wav "$cap"   # warmup primed serena -> hit unless cap=0
    req "r${round}_o1" 官方样例 wav "$cap"   # other voice -> miss
    req "r${round}_s2" serena   wav "$cap"   # KEY: cap=1 cold, cap=4 hit
    req "r${round}_o2" 官方样例 wav "$cap"   # KEY: cap=1 cold, cap=4 hit
    req "r${round}_s3" serena   wav "$cap"   # KEY
    req "r${round}_ps" serena   pcm "$cap"   # Web default path, /metrics
    $ADB shell "grep -a '\[m5\] prime' $V/m5-server.log" > "$LOGDIR/cap${cap}_r${round}.log" 2>/dev/null
done
echo "DONE" | tee -a "$OUT"
