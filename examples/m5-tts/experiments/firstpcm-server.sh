#!/usr/bin/env bash
# Server-side first-packet A/B: does the codec prime cache (restore_prime)
# survive across requests on the same voice? wav has X-M5 headers; pcm is
# chunked, so read last.first_pcm from /metrics after each request.
set -u
URL=http://127.0.0.1:18080
T1='前方三百米右转，随后靠左行驶。别着急，我们还有时间。'
T2='各位旅客请注意，开往北京南站的列车即将检票进站。'
OUT=${M5_ASSETS:-$HOME/m5-assets}/firstpcm-server-results.txt
: > "$OUT"

wav_req () {
  label=$1; voice=$2; text=$3
  curl -sS -m 90 --noproxy '*' -D /tmp/fp-h.txt -o /tmp/fp-o.wav \
    -X POST "$URL/v1/audio/speech" -H 'Content-Type: application/json' \
    -d "{\"input\":\"$text\",\"voice\":\"$voice\",\"response_format\":\"wav\"}" >/dev/null 2>&1
  fp=$(grep -ai '^X-M5-First-Pcm-S' /tmp/fp-h.txt | tr -d '\r' | awk '{print $2}')
  el=$(grep -ai '^X-M5-Elapsed-S' /tmp/fp-h.txt | tr -d '\r' | awk '{print $2}')
  fr=$(grep -ai '^X-M5-Frames' /tmp/fp-h.txt | tr -d '\r' | awk '{print $2}')
  echo "wav $label voice=$voice first=$fp elapsed=$el frames=$fr" | tee -a "$OUT"
}

pcm_req () {
  label=$1; voice=$2; text=$3
  curl -sS -m 90 --noproxy '*' -o /tmp/fp-o.pcm \
    -X POST "$URL/v1/audio/speech" -H 'Content-Type: application/json' \
    -d "{\"input\":\"$text\",\"voice\":\"$voice\",\"response_format\":\"pcm\"}" >/dev/null 2>&1
  read -r fp el fr <<<"$(curl -sS -m 6 --noproxy '*' "$URL/metrics" | python3 -c '
import sys,json
d=json.load(sys.stdin); x=d.get("last") or {}
print(x.get("first_pcm_s",0), x.get("elapsed_s",0), x.get("frames",0))')"
  echo "pcm $label voice=$voice first=$fp elapsed=$el frames=$fr" | tee -a "$OUT"
}

echo "--- wav: same voice repeated, then voice switch, then back ---" | tee -a "$OUT"
wav_req 01_serena_cold serena "$T1"
wav_req 02_serena_hit1 serena "$T1"
wav_req 03_serena_hit2 serena "$T1"
wav_req 04_serena_othertext serena "$T2"
wav_req 05_official 官方样例 "$T1"
wav_req 06_back_serena_cold serena "$T1"
wav_req 07_back_serena_hit serena "$T1"

echo "--- pcm (Web default): cold then repeated ---" | tee -a "$OUT"
pcm_req 08_pcm_cold serena "$T1"
pcm_req 09_pcm_hit1 serena "$T1"
pcm_req 10_pcm_hit2 serena "$T1"
echo DONE | tee -a "$OUT"
