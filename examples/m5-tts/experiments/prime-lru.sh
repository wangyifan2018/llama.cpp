#!/usr/bin/env bash
# Verify LRU eviction + content-based keys using the web's inline transient voices.
# Six distinct code sets are sent in an order that forces eviction of the
# least-recently-used entry at capacity 4.
set -u
ADB=${ADB:-adb}
V=/data/local/tmp/llama-m5-htp-v73
URL=http://127.0.0.1:18080
OUT=${M5_ASSETS:-$HOME/m5-assets}/prime-lru-results.txt
: > "$OUT"

M5_VOICES=${M5_VOICES:-$(dirname "$0")/voices-web.json}
export M5_VOICES
python3 - <<'PY'
import json, os
d = json.load(open(os.environ['M5_VOICES']))
codes = [v['codes'] for v in d['voices'] if v['name'] == 'serena'][0]
text = '前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
for i in range(1, 7):
    c = [row[:] for row in codes]
    c[-1][0] = (c[-1][0] + i * 37) % 2048   # distinct last-8-frame prime key
    body = {'input': text, 'voice': 'serena', 'response_format': 'wav',
            'codes': c, 'reference_text': '测试参考文本'}
    with open('/tmp/prime-lru-%d.json' % i, 'w') as f:
        json.dump(body, f, ensure_ascii=False)
PY

$ADB shell 'for p in $(pidof m5-server); do kill $p; done; sleep 2;
    for n in backbone-resident depth-resident; do
        for p in $(pidof $n 2>/dev/null); do kill $p; done
    done; sleep 1' >/dev/null 2>&1
$ADB shell "cd $V && D=\$PWD; \
    export LD_LIBRARY_PATH=\$D ADSP_LIBRARY_PATH=\$D \
           GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1 \
           M5_PRIME_CACHE=4 M5_PRIME_STATS=1; \
    nohup ./m5-server model --port 8080 --cwd \$D --threads 3 \
        --voice serena --voice-encoder \$D/codec_encoder.onnx \
        > \$D/m5-server.log 2>&1 &" >/dev/null 2>&1
for _i in $(seq 1 60); do
    curl -s -m 2 --noproxy '*' "$URL/health" | grep -q '"ok"' && { sleep 1; break; }
    sleep 1
done

# warmup(serena) + v1..v5 -> v1 evicted; v2 still resident
for seq in 1 2 3 4 5 1 2 6; do
    first=$(curl -sS -m 90 --noproxy '*' -D /tmp/plru-hdr.txt -o /tmp/plru.wav \
        -X POST "$URL/v1/audio/speech" -H 'Content-Type: application/json' \
        -d @/tmp/prime-lru-$seq.json >/dev/null 2>&1; \
        grep -ai '^X-M5-First-Pcm-S' /tmp/plru-hdr.txt | tr -d '\r' | awk '{print $2}')
    sha=$(shasum -a 256 /tmp/plru.wav | cut -c1-16)
    hits=$($ADB shell "grep -ac 'prime hit' $V/m5-server.log" | tr -d '\r')
    misses=$($ADB shell "grep -ac 'prime miss' $V/m5-server.log" | tr -d '\r')
    entries=$($ADB shell "grep -a 'prime save' $V/m5-server.log | tail -1" | grep -ao 'entries=[0-9]*')
    echo "v$seq first=$first sha=$sha cumulative_hits=$hits misses=$misses last_$entries" | tee -a "$OUT"
done
$ADB shell "grep -a '\[m5\] prime' $V/m5-server.log" > ${M5_ASSETS:-$HOME/m5-assets}/prime-lru.log
echo "DONE" | tee -a "$OUT"
